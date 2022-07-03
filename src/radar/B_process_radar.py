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
from rasterio.mask import mask as msk

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
import pyart
from rasterio.warp import \
    calculate_default_transform  # For reproject_to_file; For reproject_tif
from rasterio.warp import Resampling, reproject
from shapely.geometry import Point
import cv2
import numpy.ma as ma


sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))
from geo_utils import create_geodataframe, grow_point
from misc_utils import dates_list_from_base, get_date_from_name, nearest_date
from misc_utils import select_files_dt

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
  
# TODO import from utilities
def minmax_normalization(array, valid_min, valid_max):
    return (np.float64(array)-valid_min)/(valid_max-valid_min)

# TODO import from raster_utils
def reproject_et(inpath, outpath, new_crs):
    '''
    This function reprojects the raster to a new crs
    :param inpath: pathway to the input tif file
    :param outpath: pathway to the output tif file
    :param new_crs: The new crs
    '''
    dst_crs = new_crs
    with rio.open(inpath) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'crs': dst_crs,
            'transform': transform,
            'width': width,
            'height': height
        })
        with rio.open(outpath, 'w', **kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(source=rio.band(src, i),
                        destination=rio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.nearest)

# TODO import from raster_utils        
def grow_point(series, polygon_size, epsg, cap_style = 3):
    # polygon_size = 50000 ## Square size in meters
    series = series.to_crs(epsg)

    polygon = series['geometry'].iloc[0].buffer(polygon_size/2, cap_style = cap_style)

    polygon_df = gpd.GeoDataFrame(index=[0], crs=epsg, geometry=[polygon])
    return polygon_df

def _create_geodataframe(series):

    series['geometry'] = Point(series.longitude, series.latitude)
    series = series.to_frame().T
    try:
        zone = series.zone #hail
    except:
        zone = [item[:2] for item in list(series.mgrs.values)][0]  # non hail
    epsg = 32600 + int(zone)  ## epsg UTM North
    series = gpd.GeoDataFrame(series, crs='EPSG:4326')
    series = series.to_crs('EPSG:' + str(epsg))
    return series

def process_radar(radar_file, tif_file, array_dir, 
            field, latitude, longitude,
            # monitoring_dt
            ):
    array_file = array_dir/(tif_file.stem)
    print(f'processing {array_file}')
    if not array_file.is_file():
        array_file.parent.mkdir(parents = True, exist_ok = True)
        # read radar
        radar = pyart.io.read(radar_file)
        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 0)
        series_dic = {'latitude':latitude, 'longitude':longitude, 'zone':mgrs_coordinates[:2]}
        series = pd.Series(data=series_dic, index=['latitude', 'longitude', 'zone'])
        series = create_geodataframe(series)
        zone = series.zone
        epsg = 'EPSG:' + str(32600 + int(zone)) 
        polygon_df = grow_point(series, polygon_size=15000, epsg=epsg)

        
        try:
            # get min and max from radar
            max_value = radar.fields[field]['valid_max']
            min_value = radar.fields[field]['valid_min'] 
            
            
            with tempfile.TemporaryDirectory() as tmpdirname:
                print('created temporary directory', tmpdirname)
                # modify projected name
                # dst = os.path.join(projected_dir,f'{ICAO}_{field}_{closest_datetime.strftime("%Y%m%d_%H%M%S")}_{str(row["latitude"])}_{str(row["longitude"])}_projected.tif')
                dst = os.path.join(tmpdirname,Path(tif_file).stem + '_projected.tif') ## Edited by Naty
                
                if not Path.is_file(Path(dst)):
                    reproject_et(tif_file, dst, polygon_df.crs)
                with rio.open(dst,'r') as dst_file:
                    out_meta = dst_file.meta
                    cropped_image, out_transform = msk(dst_file, polygon_df.geometry, crop=True, all_touched=True)
                    
                
                mask = np.isnan(cropped_image)                               
                masked_image = ma.masked_array(cropped_image, mask=mask)
                masked_image[masked_image>max_value] = max_value
                masked_image[masked_image<min_value] = min_value  
                if np.sign(min_value) == -1.0:
                    fill_value = 0
                else:
                    fill_value = min_value 
                image = masked_image.filled(fill_value = fill_value)                                          

                cropped_image_norm = minmax_normalization(image, min_value, max_value) 
                image = cv2.resize(cropped_image_norm.squeeze(), (32,32) , interpolation = cv2.INTER_LINEAR)
                            


                np.save(array_file, image.data)
                del image
                del cropped_image
                del masked_image
                del cropped_image_norm
                
                success = True
        except:
            success = False
    else:
        success = True
    return success


class FileNotInConfig(Exception):
    pass


def get_details_radar2(product_dic,radar_station,raw_dir,filename):

    process_file = True
    
    fields = product_dic['fields']

    try:
        ## Selecting files based on fields of interest
        field = next(field for field in fields if field in filename.stem)
        radar_date = get_date_from_name(filename, radar_station, 8, pattern = '%Y%m%d')
        dt_dir = Path(str(radar_date.year))/radar_date.strftime('%m')/radar_date.strftime('%d')
        radar_file = raw_dir/dt_dir/radar_station/'_'.join(filename.stem.split('_')[:3])
        # /maindir/data/radar/raw/2022/05/19/KBLX/KBLX20220519_205843_V06

    except (IndexError,FileNotInConfig,StopIteration) as e:    
        process_file = False
        field = None
        radar_file = None
        logger.warning(
                'exception: %s \n%s details not included in dictionary'\
                    ' information. File will not be processed', e,filename.stem)
        

    return field, radar_file, process_file

def get_details_radar(dates_list,product_dic,radar_station,raw_dir,filename):

    process_file = True
    
    fields = product_dic['fields']

    # if any(field in filename.stem for field in fields):
    try:
        ## Checking time of file is of interest
        monitoring_date = get_date_from_name(filename, 'm')
        if monitoring_date in dates_list:
            ## Selecting files based on fields of interest
            field = next(field for field in fields if field in filename.stem)
            radar_date = get_date_from_name(filename, radar_station, 8, pattern = '%Y%m%d')
            dt_dir = Path(str(radar_date.year))/radar_date.strftime('%m')/radar_date.strftime('%d')
            radar_file = raw_dir/dt_dir/radar_station/'_'.join(filename.stem.split('_')[:3])
            # /maindir/data/radar/raw/2022/05/19/KBLX/KBLX20220519_205843_V06
            assert radar_file.is_file()
    except (IndexError,FileNotInConfig,StopIteration) as e:    
        process_file = False
        field = None
        radar_file = None
        logger.warning(
                'exception: %s \n%s details not included in dictionary'\
                    ' information. File will not be processed', e,filename.stem)
        

    return field, radar_file, process_file


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
    parser.add_argument('-rad','--radar',
                    default='KBLX',
                    help='5 letter code for radar station')
    parser.add_argument('-c','--country',
                    default='US',
                    choices=['US', 'CA'],
                    help='Country of radar station')    

    config = parser.parse_args()
    config = vars(config)
    latitude = float(config['latitude'])
    longitude = float(config['longitude'])
    # dt = datetime.now() - timedelta(minutes = 60)
    
    dt = datetime(int(config['year']), int(config['month']), int(config['day']), int(config['hour']), 0, 0, 0)
    radar_station = config['radar']
    country = config['country']

    # TODO Add argparse debug
    # for hour in range(1,24):

    debug = False
    if debug:
        dt = datetime(2020,7,7,0)
        dt = datetime(2022,5,28,10)
        dt = datetime(2020,6,4,12)
    #     dt = dt + timedelta(hours= hour)
    config_file = CONFIG_DIR/'config_radar.json'
    with open(config_file) as f:
        product_dic = json.load(f)  

    old = False
    if not old:
        minutes_subtract = 60
        minutes_frequency = 6

        
        dt = dt - timedelta(minutes = minutes_subtract)
        grid_dir = DATA_DIR/'radar/grid'
        raw_dir = DATA_DIR/'radar/raw'

        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
        area = f'area_{mgrs_coordinates}'

        array_dir = DATA_DIR/Path('radar/np_arrays')
        array_dir = array_dir/area

        file_list = []
        for delta_t in range(-120, 1, 60):
            dt_process = dt + timedelta(minutes = delta_t)
            dt_dir = Path(str(dt_process.year))/dt_process.strftime('%m')/dt_process.strftime('%d')/dt_process.strftime('%H')
            grid_dt_dir = grid_dir/dt_dir/radar_station
            file_list.append(sorted(grid_dt_dir.rglob('*.tif')))

        file_list = [item for sublist in file_list for item in sublist]
        
        files_df = select_files_dt(file_list, dt-timedelta(hours = 1), minutes_frequency)

        for unique_dt in files_df['date'].unique():
            unique_dt_dir = str(unique_dt.astype('datetime64[h]')).replace('-','/').replace('T','/')
            array_dir_process = array_dir/unique_dt_dir/radar_station
            files_dt_df = files_df[files_df['date'] == unique_dt]

            for _, row in files_dt_df.iterrows():
                tif_file = row['file']
                field, radar_file, process_file = get_details_radar2(product_dic,radar_station,raw_dir,tif_file)
                if process_file:
                    process_radar(radar_file, tif_file, array_dir_process, field, latitude, longitude)
         
    else:
        minutes_subtract = 0
        minutes_frequency = 6

        dt_process = dt - timedelta(minutes = minutes_subtract)
        dt_dir = Path(str(dt_process.year))/dt_process.strftime('%m')/dt_process.strftime('%d')/dt_process.strftime('%H')
        
        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
        area = f'area_{mgrs_coordinates}'

        array_dir = DATA_DIR/Path('radar/np_arrays')
        array_dir = array_dir/area/dt_dir/radar_station
        # print(array_dir)
        
        grid_dir = DATA_DIR/'radar/grid'/dt_dir/radar_station
        raw_dir = DATA_DIR/'radar/raw'
        dates_list = dates_list_from_base(dt_process, minutes_subtract+1, minutes_frequency)

        for tif_file in grid_dir.rglob('*.tif'):
            field, radar_file, process_file = get_details_radar(dates_list,product_dic,radar_station,raw_dir,tif_file)
            if process_file:
                process_radar(radar_file, tif_file, array_dir, field, latitude, longitude)
            
        minutes_subtract = 60
        minutes_frequency = 6
        dt_process = dt - timedelta(minutes = minutes_subtract)
        dt_dir = Path(str(dt_process.year))/dt_process.strftime('%m')/dt_process.strftime('%d')/dt_process.strftime('%H')
        
        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
        area = f'area_{mgrs_coordinates}'

        array_dir = DATA_DIR/Path('radar/np_arrays')
        array_dir = array_dir/area/dt_dir/radar_station
        # print(array_dir)
        
        grid_dir = DATA_DIR/'radar/grid'/dt_dir/radar_station
        raw_dir = DATA_DIR/'radar/raw'
        dates_list = dates_list_from_base(dt_process, minutes_subtract, minutes_frequency)
                

        for tif_file in grid_dir.rglob('*.tif'):
            field, radar_file, process_file = get_details_radar(dates_list,product_dic,radar_station,raw_dir,tif_file)
            if process_file:
                process_radar(radar_file, tif_file, array_dir, field, latitude, longitude)
                    

        dt = dt - timedelta(hours = 1)
        dt_process = dt - timedelta(minutes = minutes_subtract)
        dt_dir = Path(str(dt_process.year))/dt_process.strftime('%m')/dt_process.strftime('%d')/dt_process.strftime('%H')
        
        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
        area = f'area_{mgrs_coordinates}'

        array_dir = DATA_DIR/Path('radar/np_arrays')
        array_dir = array_dir/area/dt_dir/radar_station
        # print(array_dir)
        
        grid_dir = DATA_DIR/'radar/grid'/dt_dir/radar_station
        raw_dir = DATA_DIR/'radar/raw'
        dates_list = dates_list_from_base(dt_process, minutes_subtract, minutes_frequency)
                

        for tif_file in grid_dir.rglob('*.tif'):
            field, radar_file, process_file = get_details_radar(dates_list,product_dic,radar_station,raw_dir,tif_file)
            if process_file:
                process_radar(radar_file, tif_file, array_dir, field, latitude, longitude)
    
    
    
    
    # # 3 bands
    # product_dic =  {'ABI-L2-CMIPC':{'name':'CMI', 'channels':[f'C{i:02d}' for i in range(1,2)]},
    #     'ABI-L2-ACHTF':{'name':'TEMP', 'channels':['']}, 
    #     'ABI-L2-ACTPC':{'name':'Phase', 'channels':['']}
    # }

    print('end')
    


if __name__ == '__main__':
    main()

# array_name = 'GOES_s' + goes_dt.strftime('%Y%j_%H%M%S') +'_h' + dt.strftime('%Y%j_%H%M%S') + '_' + str(row['latitude']) + '_' + str(row['longitude'])
            