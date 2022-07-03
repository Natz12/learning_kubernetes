import argparse
import json
import logging
import os
import resource
import shutil
import sys
import tempfile
import time
import warnings
from asyncio import events
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from itertools import repeat
from logging import handlers
from pathlib import Path

import cv2
import geopandas as gpd
import google.cloud.storage as gcs
import matplotlib.pyplot as plt
import mgrs
import netCDF4 as nc
import numpy as np
import numpy.ma as ma
import pandas as pd
import rasterio as rio
import rioxarray
import torchvision.models as models
import torchvision.transforms as transforms
from osgeo import gdal
from rasterio.mask import mask
from scipy import stats
from shapely.geometry import Point, Polygon

sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))
from geo_utils import create_geodataframe, grow_point
from misc_utils import dates_list_from_base, get_goes_start_date, nearest_date,get_goes_date

warnings.filterwarnings("ignore")

USE_PARALLEL = True

DATA_DIR = Path(__file__).resolve().parent.parent/'data'
CONFIG_DIR = Path(__file__).resolve().parent.parent/'config'

LOG_DIR = DATA_DIR/"logs"

LOG_FILE = True

logging.root.handlers = []


def get_logger(LOG_FILE):
    """_summary_

    Args:
        LOG_FILE (_type_): _description_

    Returns:
        _type_: _description_
    """
    log = logging.getLogger('myLogger')
    if len(log.handlers) == 0:
        log.setLevel(logging.WARNING)
        log_format = logging.Formatter(
            '[%(asctime)s - %(filename)s:%(lineno)s - %(funcName)s() - %(levelname)s] %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(log_format)
        log.addHandler(ch)
        if LOG_FILE:
            fh = handlers.RotatingFileHandler(LOG_DIR / "goes_process.log", maxBytes=(1048576 * 5), backupCount=7)
            fh.setFormatter(log_format)
            log.addHandler(fh)
    return log


logger = get_logger(LOG_FILE)

# Limit memory use
# From https://stackoverflow.com/questions/39755928/how-do-i-use-setrlimit-to-limit-memory-usage-rlimit-as-kills-too-soon-rlimit
# And https://stackoverflow.com/a/33525161
rsrc = resource.RLIMIT_AS
soft, hard = resource.getrlimit(rsrc)
logger.info("Memory limit starts as: {0}, {1}".format(soft, hard))
size = 13 * 1024 * 1024 * 1024  # In bytes
resource.setrlimit(rsrc, (size, size))
logger.info("Memory limit set to: {0}, {1}".format(soft, hard))



# 44.06418822, -103.41288591


def resample(image, width, height, interpolation = 'NEAREST'):
    """
    Resample an image to the specified width and height, using the specified interpolation method.
    Args:
        layer: The image to resample
        width: Desired width
        height: Desired height.
        interpolation: Interpolation method, as a string. The options are the same as of OpenCV:
            INTER_NEAREST – a nearest-neighbor interpolation: not so good for enlarging
            INTER_LINEAR – a bilinear interpolation
            INTER_AREA – resampling using pixel area relation. It may be a preferred method for image decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to the INTER_NEAREST method.
            INTER_CUBIC – a bicubic interpolation over 4×4 pixel neighborhood
            INTER_LANCZOS4 – a Lanczos interpolation over 8×8 pixel neighborhood
    Returns:
        numpy.ma: The resampled image, as a numpy masked array
    """
    import cv2
    import numpy.ma as ma
    if isinstance(interpolation, str):
        if interpolation == 'NEAREST' or interpolation == 'INTER_NEAREST':
            interpol_method = cv2.INTER_NEAREST
        elif interpolation == 'LINEAR' or interpolation == 'INTER_LINEAR':
            interpol_method = cv2.INTER_LINEAR
        elif interpolation == 'AREA' or interpolation == 'INTER_AREA':
            interpol_method = cv2.INTER_AREA
        elif interpolation == 'CUBIC' or interpolation == 'INTER_CUBIC':
            interpol_method = cv2.INTER_CUBIC
        elif interpolation == 'LANCZOS4' or interpolation == 'INTER_LANCZOS4':
            interpol_method = cv2.INTER_LANCZOS4
        else:
            raise ValueError
    else:
        raise TypeError
    
    img = ma.masked_array(image)
    img_fill = img.fill_value
    img = img.filled()
    dim = (width, height)
    # resize image
    resampled = cv2.resize(img, dim, interpolation = interpol_method)
    resampled = ma.masked_values(resampled, img_fill)
    return resampled  

    
def minmax_normalization(array, product, valid_range):
    """_summary_

    Args:
        array (_type_): _description_
        product (_type_): _description_
        valid_range (_type_): _description_

    Returns:
        _type_: _description_
    """
    mask = [array == -1]
    masked_image = ma.masked_array(array, mask=mask)
    minmax_dict = {'ABI-L2-ACHTF':(0, 350), 'ABI-L2-ACTPC':(1, 4), 'ABI-L2-CMIPC':(-1, 1.5)}
    if 'ABI-L2-ACHTF' in product:
        norm_array = (masked_image-minmax_dict['ABI-L2-ACHTF'][0])/(minmax_dict['ABI-L2-ACHTF'][1]-minmax_dict['ABI-L2-ACHTF'][0])
    else:
        norm_array = (masked_image-valid_range[0])/(valid_range[1]-valid_range[0])
    return norm_array

def process_goes(nc_file, array_dir, product, name, latitude, longitude, monitoring_dt):
    """_summary_

    Args:
        nc_file (_type_): _description_
        product (_type_): _description_
        name (_type_): _description_
        latitude (_type_): _description_
        longitude (_type_): _description_

    Returns:
        _type_: _description_
    """
    tif_file = nc_file.with_suffix('.tif')
    tif_file = Path(str(tif_file).replace('raw_nc','raw'))
    try:
        with rioxarray.open_rasterio(nc_file) as s:
            file_scale_factor = s[name].scale_factor
            file_offset = s[name].add_offset
            valid_range = s[name].valid_range
        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 0)
        series_dic = {'latitude':latitude, 'longitude':longitude, 'zone':mgrs_coordinates[:2]}
        series = pd.Series(data=series_dic, index=['latitude', 'longitude', 'zone'])
        series = create_geodataframe(series)
        zone = series.zone
        epsg = 'EPSG:' + str(32600 + int(zone))  ## epsg UTM North
        polygon_big_df = grow_point(series, square_size=100000, epsg=epsg)

        # HERE
        ## Cropping to big polygon
        

        with tempfile.TemporaryDirectory() as tmpdirname:
            print('created temporary directory', tmpdirname)
        # directory and contents have been removed

            with rio.open(tif_file) as src:
                out_meta = src.meta
                polygon_big_df = polygon_big_df.to_crs(out_meta['crs'])
                out_image, out_transform = mask(src, polygon_big_df.geometry, crop=True, all_touched=True)
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})
            # dst = os.path.join(out_dir, 'files/'+str(file_name[:-3]) + "_" + product + '_big' + '.tif')
            
            polygon_big_file = os.path.join(tmpdirname,Path(tif_file).stem + '.tif') ## Edited by Naty
            with rio.open(polygon_big_file, "w", **out_meta) as dst:
                dst.write(out_image)
            ## Cropping to small polygon
            epsg = out_meta['crs']
            polygon_df = grow_point(series, square_size=15000, epsg=epsg)
            with rio.open(polygon_big_file) as src:
                out_meta = src.meta
                # TODO Assert extend of polygon_df is inside extend of out_meta['crs]. If not, then polygon_df will become infinite
                polygon_df = polygon_df.to_crs(out_meta['crs'])
                out_image, out_transform = mask(src, polygon_df.geometry, crop=True, all_touched=True)
            out_meta.update({"driver": "GTiff",
                            "height": out_image.shape[1],
                            "width": out_image.shape[2],
                            "transform": out_transform})
            
            
            # dst = os.path.join(raw_dir, Path(str(file_name)).stem + "_" + product + '_crop' + '.tif') ## Edited by Naty
            # with rio.open(dst, "w", **out_meta) as dest:
            #     dest.write(out_image)
            
            out_image = out_image.squeeze().astype("float32", casting="unsafe")
            mask1 = [out_image == -1]
            masked_image = ma.masked_array(out_image, mask=mask1)
            masked_image *= file_scale_factor
            masked_image += file_offset
            valid_range = valid_range*file_scale_factor+file_offset
            
            # out_image = masked_image.data
            
            
            # nan_occurrences = np.count_nonzero(out_image == -1)
            # if nan_occurrences < 0.5*(out_image.size):
                ## fill and NORMALIZE THE ARRAY
            # array = filling(array=out_image,nan_value=-1,axis_index=0,product=product)
            # array = filling(array=array,nan_value=0,axis_index=1,product=product)
            
            
            # array = resample(masked_image, 128, 128, interpolation='LINEAR')
            
            array = cv2.resize(masked_image, (32,32) , interpolation = cv2.INTER_LINEAR)
            # if 'CMIPC' not in product: # CMIPC already between 0 and 1
            array = minmax_normalization(array,product,valid_range)

            array_file = array_dir/(tif_file.stem +'_m'+ datetime.strftime(monitoring_dt,'%Y%j%H%M'))
            array_file.parent.mkdir(parents = True, exist_ok = True)

            np.save(array_file, array.data)
            del out_image
            del polygon_big_df
            del polygon_df
            del mask1
            del masked_image
            
            success = True
    except:
        success = False
        # print("nc_file, array_dir, product, name, latitude, longitude, monitoring_dt",
        #     nc_file, array_dir, product, name, latitude, longitude, monitoring_dt)
    
    return array, success

class FileNotInConfig(Exception):
    pass

def get_details_goes(product_dic,file_name):
    """_summary_

    Args:
        product_dic (_type_): _description_
        file_name (_type_): _description_

    Raises:
        FileNotInConfig: _description_

    Returns:
        _type_: _description_
    """
    process_file = True
    prod_dic = {outer_k: inner_v for outer_k, outer_v in \
            product_dic.items() for inner_k, inner_v in \
            outer_v.items() if 'channels' not in inner_k}
    product = file_name.stem.split('_')[1]
    name = [val for key, val in prod_dic.items() if key in product]
    try:
        name = name[0]
        if name == 'CMI':
            prod_dic = {key:value for key,value in product_dic.items() if name in key}
            channels = list({key:value for key,value in product_dic.items() if name in key}.values())[0]['channels']
            if not any(substring in file_name.stem for substring in channels):
                raise FileNotInConfig
    except (IndexError,FileNotInConfig) as e:
        process_file = False
        logger.warning(
                'exception: %s \n%s details not included in dictionary'\
                    ' information. File will not be processed', e,file_name.stem)
        

    return product, name, process_file


def main():
    """_summary_
    """

    # latitude = 46.511909
    # longitude = -109.156326
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-yr','--year',
                    default=str(datetime.now().year),
                    help='4-digit year to be processed')
    parser.add_argument('-m','--month',
                    default=str(datetime.now().month),
                    help='1 or 2 digit month to be processed')
    parser.add_argument('-d','--day',
                    default=str(datetime.now().day),
                    help='1 or 2 digit day to be processed')
    parser.add_argument('-hr','--hour',
                    default=str(datetime.now().hour),
                    help='1 or 2 digit hour. Hour-1 will be processed')
    parser.add_argument('-lat','--latitude',
                    default=46.511909,
                    help='float representing the latitude of area to be processed')
    parser.add_argument('-lon','--longitude',
                    default=-109.156326,
                    help='float representing the longitude of area to be '\
                        ' processed. Probably negative.')
    
    config = parser.parse_args()
    config = vars(config)
    latitude = float(config['latitude'])
    longitude = float(config['longitude'])
    # dt = datetime.now() - timedelta(minutes = 60)
    
    dt = datetime(int(config['year']), int(config['month']), int(config['day']), int(config['hour']), 0, 0, 0)


    debug = True
    # TODO Add argparse debug
    # for hour in range(1,24):
    if debug:
        dt = datetime(2020,7,7,23)
        dt = datetime(2020,7,8,0)
        dt = datetime(2020,6,4,11)
    #     dt = dt + timedelta(hours= hour)

    minutes_subtract = 60
    minutes_frequency = 10
    dt = dt - timedelta(minutes = minutes_subtract)
    dt_dir = Path(str(dt.year))/dt.strftime('%m')/dt.strftime('%d')/dt.strftime('%H')
    
    m = mgrs.MGRS()
    mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
    area = f'area_{mgrs_coordinates}'

    array_dir = DATA_DIR/Path('goes/np_arrays')
    array_dir = array_dir/area/dt_dir
    # print(array_dir)
    
    raw_dir = DATA_DIR/'goes/raw_nc'/dt_dir
    dates_list = dates_list_from_base(dt, minutes_subtract, minutes_frequency)
    
    # product_dic = {outer_k: {inner_k: inner_v for inner_k, inner_v in \
    #         outer_v.items() if 'channels' not in inner_k} for outer_k, outer_v in \
    #         product_dic.items()}

    config_file = CONFIG_DIR/'config_goes.json'

    with open(config_file) as f:
        product_dic = json.load(f)          


    for nc_file in raw_dir.rglob('*.nc'):
        product, name, process_file = get_details_goes(product_dic,nc_file)
        goes_dt = get_goes_date(nc_file, 's')
        monitoring_dt = nearest_date(dates_list, goes_dt)
        if process_file:
            process_goes(nc_file, array_dir, product, name, latitude, longitude, monitoring_dt)
        
    
    # # 3 bands
    # product_dic =  {'ABI-L2-CMIPC':{'name':'CMI', 'channels':[f'C{i:02d}' for i in range(1,2)]},
    #     'ABI-L2-ACHTF':{'name':'TEMP', 'channels':['']}, 
    #     'ABI-L2-ACTPC':{'name':'Phase', 'channels':['']}
    # }

    print('end')
    


if __name__ == '__main__':
    main()

# array_name = 'GOES_s' + goes_dt.strftime('%Y%j_%H%M%S') +'_h' + dt.strftime('%Y%j_%H%M%S') + '_' + str(row['latitude']) + '_' + str(row['longitude'])
            