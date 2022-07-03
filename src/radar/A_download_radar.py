import argparse
import contextlib
import gc
import json
import logging
import os
import resource
import sys
import time
import warnings
from datetime import datetime, timedelta
from logging import handlers
from pathlib import Path

import cv2
import geopandas as gpd
import google.cloud.storage as gcs
import numpy as np
import numpy.ma as ma
import pandas as pd
import pyart
import rasterio as rio
import rioxarray
from boto.s3.connection import S3Connection

sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))
from geo_utils import create_geodataframe, grow_point

warnings.filterwarnings("ignore")

USE_PARALLEL = True

DATA_DIR = Path(__file__).resolve().parent.parent/'data'
CONFIG_DIR = Path(__file__).resolve().parent.parent/'config'

LOG_DIR = DATA_DIR/"logs"

LOG_FILE = True

logging.root.handlers = []


def get_logger(log_file):
    """_summary_

    Args:
        log_file (_type_): _description_

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
        if log_file:
            fh = handlers.RotatingFileHandler(LOG_DIR / "goes_download.log", maxBytes=(1048576 * 5), backupCount=7)
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

try:
    from joblib import Parallel, parallel

    parallel_available = True
    max_cpu = int(os.cpu_count() * 3 / 4)
    max_cpu = 4
    num_workers = max_cpu  # Number of workers to use. -1 is all, -2 is all but 1...
    verbose = 10  # Level of verbose to print for the joblib parallel processes
    logger.info('Parallelization available, setting max number of CPUs to use to {}'.format(max_cpu))
except ImportError as e:
    parallel_available = False
    logger.warning('Parallelization not available, joblib library could not be loaded')

def create_dir(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)



# # Add a decorator that will make timer() a context manager
# @contextlib.contextmanager
# def timer():
#   """Time the execution of a context block.

#   Yields:
#     None
#   """
#   start = time.time()
#   # Send control back to the context block
#   yield
#   end = time.time()
#   print('Elapsed: {:.2f}s'.format(end - start))

# A Context manager to catch performance timing
# https://stackoverflow.com/a/33987224


def save_tif(radar:pyart.core.Radar, fields, out_dir: str, file_name:str, resolution:int = 1000, **kwargs):
    """
    Takes the field of radar and translates and saves it into a geotiff in the specified directory
    Args:
        radar: pyart.core.radar.Radar radar object to be translated into geotiff
        fields: fields to be exported as Geotiff
        out_dir: output directory
        resolution: output resolution in meters per pixel
    Returns:
        Nothing
    """
    ## Standar grid (Default)
    ### WARNING
    # @striges Yes, Py-ART's gridding method should only be used to combine scalar 
    # fields from more than one radar, it will give poor results for vector or 
    # phase fields. Insuring that only scalar fields are gridded is left up to the 
    # user, it is difficult to determine if a field is a scalar, vector, phase, etc
    # from that data in a Radar object, but the users should (hopefully) know this 
    # information. The plan is to include routines that can deal with doppler 
    # velocities soon and any contribution along these lines would be greatly 
    # appreciated.
    # https://github.com/ARM-DOE/pyart/issues/146

    max_range = radar.range['data'].max() + 100000
    shape_xy = int(np.ceil(max_range/resolution))
    grid = pyart.map.grid_from_radars(radar, 
        # gatefilters=(gatefilter),
        grid_shape = (1, shape_xy, shape_xy),
        grid_limits = ((0,radar.gate_altitude['data'].max()),
                    (-shape_xy*resolution/2, shape_xy*resolution/2),
                    (-shape_xy*resolution/2, shape_xy*resolution/2)),
        weighting_function = "BARNES2",
        #              (rad_coord_cart[1]-max_range, rad_coord_cart[1]+max_range),
        #              (rad_coord_cart[0]-max_range, rad_coord_cart[0]+max_range)),
        # Zone 12N Projected Bounds: 196765.3486, 2749459.8086, 803234.6514, 8799482.7282

        fields = fields)
    for field in fields:
        f_name = out_dir + '/' + file_name + '_' + field
        # pyart.io.write_grid_geotiff(grid, f_name, field, warp = True, sld=True, use_doublequotes= True)
        pyart.io.write_grid_geotiff(grid, f_name, field, warp = False, sld=True, use_doublequotes= True)
        # hail2 has warp = False


def get_radar(curr_dt:datetime, radar_station:str, data_dir:str, product_dict:dict):
    # radar_station:str, curr_dt:datetime, out_dir:str, fields:Union[list,array] = [ 'reflectivity', 'differential_reflectivity', 'cross_correlation_ratio']):
    """
    Downloads pyart.radar specified by the radar station and time
    Args:
        row: dataframe row with information about Radar station name and datetime to query radar data
        out_dir: output directory
    Returns:
        success: Bool indicating if download was successful
    """

    def _radar_to_grid(sweep, radar, field, algorithm = 'Barnes2'):

        resolution = 500
        max_range = radar.range['data'].max()
        shape_xy = int(np.ceil(2*max_range/resolution))
        grid = pyart.map.grid_from_radars(sweep, 
            # gatefilters=(gatefilter),
            grid_shape=(1, shape_xy, shape_xy),
            # grid_limits=((2000, 2000), (-523000.0, 523000.0), (-123000.0, 923000.0)),                                  
            grid_limits = ((0,radar.gate_altitude['data'].max()),
                            # (-shape_xy*resolution, shape_xy*resolution),
                            # (-shape_xy*resolution, shape_xy*resolution)),
                            (-shape_xy*resolution/2, shape_xy*resolution/2),
                            (-shape_xy*resolution/2, shape_xy*resolution/2)),
            # Zone 12N Projected Bounds: 196765.3486, 2749459.8086, 803234.6514, 8799482.7282
            weighting_function = algorithm, #Functions used to weight nearby collected points when interpolating a grid point.
            # grid_projection = proj_params,
            fields=[field]) 
        return grid

    
    
    # The underscore prefix is meant as a hint to another programmer that 
    # a variable or method starting with a single underscore is intended for internal use
    def _nearestDate(dates, pivot):
        return min(dates, key=lambda x: abs(x - pivot))
    
    def _get_closest_nexrad_time(curr_dt, radar_station):
        my_pref = curr_dt.strftime('%Y/%m/%d/') + radar_station
        conn = S3Connection(anon=True)
        bucket = conn.get_bucket('noaa-nexrad-level2')
        bucket_list = list(bucket.list(prefix=my_pref))
        if not bucket_list:
            raise FileNotFoundError('Empty bucket list.')
        keys = []
        datetimes = []
        dt = curr_dt
        for i in range(len(bucket_list)):
            this_str = str(bucket_list[i].key)
            if 'gz' in this_str:
                try:
                    endme = this_str[-22:-4]
                    fmt = '%Y%m%d_%H%M%S_V0'
                    dt = datetime.strptime(endme, fmt)
                except ValueError:
                    endme = this_str[-18:-3]
                    fmt = '%Y%m%d_%H%M%S'
                    dt = datetime.strptime(endme, fmt)
                datetimes.append(dt)
                keys.append(bucket_list[i])
            if this_str[-3::] == 'V06':
                endme = this_str[-19::]
                fmt = '%Y%m%d_%H%M%S_V06'
                dt = datetime.strptime(endme, fmt)
                datetimes.append(dt)
                keys.append(bucket_list[i])
        closest_datetime = _nearestDate(datetimes, curr_dt)

        return closest_datetime,datetimes,keys
    
    def _download_radar(closest_datetime, datetimes, keys, raw_dir):
        
        tries_get_radar = 0
        # Q: Do we remember why we tried 3 times?
        while tries_get_radar<3:
            tries_get_radar += 1
            index = datetimes.index(closest_datetime)
            
            # with tempfile.NamedTemporaryFile() as localfile:
            radar_file = raw_dir/keys[index].name
            radar_file.parent.mkdir(parents=True, exist_ok=True)
            if not radar_file.is_file():
                keys[index].get_contents_to_filename(radar_file)
            try:
                radar = pyart.io.read(radar_file)
                success = True
                break
            except:
                success = False

        return radar, radar_file, success
    
    def _get_radar_data(download_dt, radar_station,
                            data_dir, product_dict):
        raw_dir = data_dir/'raw'
        grid_dir = data_dir/'grid'
        logger = get_logger(LOG_FILE)
        # if not Path.is_file(Path(out_dir)/str(filename+'.npy')):
        fields = product_dict['fields'].copy()
        closest_datetime, datetimes, keys = _get_closest_nexrad_time(download_dt, radar_station)
        
        radar, radar_file, success_download = _download_radar(closest_datetime, datetimes, keys, raw_dir)
            
            # azimuth = sweep.azimuth['data']
            # fields = list(param.values())[0]
            # fields = param['fields'].copy()
            # slice_angle = param['slice_angle']
            # dim = (720, 1832)
            # arrays = []  
            # reset_sweep = False 
        dt_dir = Path(str(download_dt.year))/download_dt.strftime('%m')/download_dt.strftime('%d')/download_dt.strftime('%H')
        
        while fields and success_download:
            field = fields.pop(0)
            tries = 2
            tif_file = grid_dir/dt_dir/radar_station
            # tif_file = Path(str(radar_file.parent).replace('raw','grid'))
            download_dt_str = datetime.strftime(download_dt,'%Y%j%H%M')
            tif_file = tif_file/(radar_file.stem+f'_m{download_dt_str}_{field}.tif')
            sweep_num = 0
            if not tif_file.is_file():
                tif_file.parent.mkdir(parents= True, exist_ok = True)
                print(tif_file)
                while tries > 0:
                    try:                    
                        sweep = radar.extract_sweeps([sweep_num]) 
                        if sweep.fixed_angle['data'] > 0.54:
                            tries -= 1
                            sweep_num += 1
                            success_download = False
                            continue

                        field_array = sweep.fields[field]['data']

                        if field_array.mask.all():
                            tries -= 1
                            sweep_num += 1
                            success_download = False
                            continue

                        grid = _radar_to_grid(sweep, radar, field)                                            

                        pyart.io.write_grid_geotiff(grid, str(tif_file), field,
                            warp = True, sld=False, use_doublequotes= True)
                        success_download = True
                        break
                    except:
                        tries -= 1
                        sweep_num += 1
                        success_download = False
                        continue
            
            else: 
                logger = get_logger(LOG_FILE)
                logger.info('Found file: {} already on disk'.format(tif_file))
            

        return success_download
    
    ## for moving raw/projected files to s3 simultaneously
    def run_aws_command(aws_cmd, TIMEOUT):
        aws_operation = aws_cmd[2]
        success = False
        # try:
        #     completed_process = subprocess.run(aws_cmd,
        #                                         timeout = TIMEOUT,
        #                                         )
        #     if completed_process.stdout is not None:
        #         logger.info(f'STDOUT: {completed_process.stdout}')
        #     if completed_process.stderr is not None:
        #         logger.info(f'STDERR: {completed_process.stderr}')
        #     logger.info(f'AWS {aws_operation} performed successfully')
        # except subprocess.TimeoutExpired:
        #     logger.warning(f'Timeout expired for running AWS {aws_operation}')
        #     completed_process = None
        aws_operation_cmd = ' '.join(aws_cmd)
        try: 
            os.system(aws_operation_cmd)
            success = True
        except:
            logger.warning(f'Timeout expired for running AWS {aws_operation}')
        return success

    def run_aws_operation(aws_operation, AWS_PATH, SOURCE_PATH, destination_path, AWS_PROFILE, SSE_KEY_ID, KMS, TIMEOUT, mv_dt, logger):
        # logger = get_logger()
        aws_cmd =  [AWS_PATH, 's3', aws_operation, SOURCE_PATH, destination_path, '--profile', AWS_PROFILE, f'--recursive --exclude "*{mv_dt}*"']
        if KMS:
            aws_cmd.append['--sse', 'aws:kms', '--sse-kms-key-id', SSE_KEY_ID]
        logger.info(f'Running AWS {aws_operation}')
        completed_process = run_aws_command(aws_cmd, TIMEOUT)
        return completed_process
    ##
    # create_dir(out_dir)
   
    # filename=f'NEXRAD_{radar_station}_{curr_dt.strftime("%Y%m%d_%H%M%S")}_{str(row["latitude"])}_{str(row["longitude"])}'
    # if not Path.is_file(Path(out_dir)/str(filename+'.npy')):
    # if True:

    
    for delta_t in range(-60, 1, 6):
        download_dt = curr_dt + timedelta(minutes=delta_t)
        success = _get_radar_data(download_dt, radar_station,
                        data_dir, product_dict)
    
    # # for stack
    #     time_arrays.append(array_stack)

    # if success:
    #     time_stack = np.stack(time_arrays)
    #     if not time_series:
    #         time_stack= np.concatenate(time_stack)
    #     array_dir = os.path.join(out_dir,f'{filename}.npy')
    #     # Path(array_dir.parent).mkdir(parents=True, exist_ok=True)
    #     np.save(array_dir, time_stack)
            
        
        
    return success



      




def main():
    """_summary_
    """
    raw_dir = DATA_DIR/'radar'

    parser = argparse.ArgumentParser()
    parser.add_argument('-yr','--year',
                    default=str(datetime.now().year),
                    help='4-digit year to be downloaded')
    parser.add_argument('-m','--month',
                    default=str(datetime.now().month),
                    help='1 or 2 digit month to be downloaded')
    parser.add_argument('-d','--day',
                    default=str(datetime.now().day),
                    help='1 or 2 digit day to be downloaded')
    parser.add_argument('-hr','--hour',
                    default=str(datetime.now().hour),
                    help='1 or 2 digit hour. Hour-1 will be downloaded')
    parser.add_argument('-rad','--radar',
                    default='KBLX',
                    help='5 letter code for radar station')
    parser.add_argument('-c','--country',
                    default='US',
                    choices=['US', 'CA'],
                    help='Country of radar station')
    
    config = parser.parse_args()
    config = vars(config)

    # dt = datetime.now() - timedelta(minutes = 60)
    
    dt = datetime(int(config['year']), int(config['month']), int(config['day']), int(config['hour']), 0, 0, 0)
    
    radar_station = config['radar']
    country = config['country']
    # KBLX
    # KTFX
    config_file = CONFIG_DIR/'config_radar.json'

    # TODO Add argparse debug
    debug = False
    if debug:
        dt = datetime(2022,5,28,10)
        # for hour in range(1,25):
        dt = datetime(2020,6,4,12)

        # dt = dt + timedelta(hours = hour)
        with open(config_file) as f:
            product_dic = json.load(f)
        # hour X
        dt = dt - timedelta(hours = 1) # Hour X-1
        print(dt)
        get_radar(dt,
                radar_station,  
                raw_dir, 
                product_dic) #Downloading data from X-2 to X-1
        
        dt = dt - timedelta(hours = 1) # Hour X-2
        get_radar(dt,
                radar_station,  
                raw_dir, 
                product_dic) #Downloading data from X-3 to X-2
    # 

    else:
    # params =  {'fields' : ['reflectivity','differential_reflectivity', 
    #             'cross_correlation_ratio', 'velocity', 'spectrum_width', 'differential_phase'], 
    #         "kms": 15000}

    # with open(config_file, 'w') as outfile:
    #     json.dump(params, outfile)

        with open(config_file) as f:
            product_dic = json.load(f)
        dt = dt - timedelta(hours = 1)
        print(dt)
        get_radar(dt,
                radar_station,  
                raw_dir, 
                product_dic)
        
        dt = dt - timedelta(hours = 1)
        get_radar(dt,
                radar_station,  
                raw_dir, 
                product_dic)
    

    # # 3 bands
    # products = {'ABI-L2-CMIPC':{'name':'CMI', 'channels':[f'C{i:02d}' for i in range(1,2)]},
    #             'ABI-L2-ACHTF':{'name':'TEMP', 'channels':['']}, 
    #             'ABI-L2-ACTPC':{'name':'Phase', 'channels':['']}
    # }
    # get_goes(dt=dt, raw_dir=raw_dir, product_dict = products)

    print('end')

if __name__ == '__main__':
    main()
