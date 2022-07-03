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

import google.cloud.storage as gcs
import numpy as np
import pandas as pd
import rioxarray

sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))
from geo_utils import create_geodataframe, grow_point

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
except:
    parallel_available = False
    logger.warning('Parallelization not available, joblib library could not be loaded')

def create_dir(out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

def get_goes(dt, raw_dir, product_dict):

    """
    saves goes data as geotiff specified by the variables and date dt.

    Args:
        variables: str
            goes variables/bands to save
        dt: datetime
            Datetime.datetime to query goes data

    Returns:
        Success (Bool)

    """
    success = False
    logger = get_logger(LOG_FILE)

    def _nearest_date(dates, pivot):
        """Finds nearest date to pivot

        Args:
            dates (_type_): _description_
            pivot (_type_): _description_

        Returns:
            _type_: _description_
        """
        return min(dates, key=lambda x: abs(x - pivot))
    
    def get_object_id_at(dt, products='ABI-L1b-RadF', channel='C14'):
        """
        Gets GOES object ID from Google Cloud Services

        Args:
            dt: datetime
                Date for request
            product: str
                GOES product name
            channel: str
                product channel to return

        Returns:
            Object Id
        """

        # get first 11-micron band (C14) at this hour
        # See: https://www.goes-r.gov/education/ABI-bands-quick-info.html
        logger = get_logger(LOG_FILE)
        logger.info('Looking for data collected on {}'.format(dt))
        dayno = dt.timetuple().tm_yday
        gcs_prefix = '{}/{}/{:03d}/{:02d}/'.format(products, dt.year, dayno, dt.hour)
        gcs_patterns = [channel,
                        's{}{:03d}{:02d}'.format(dt.year, dayno, dt.hour)]
        blobs = list_gcs(GOES_PUBLIC_BUCKET, gcs_prefix, gcs_patterns)
        if len(blobs) > 0:
            dt_list = [datetime.strptime(blob.name.split('_')[3][1:-1], '%Y%j%H%M%S') for blob in blobs]
            dl_dt  = _nearest_date(dt_list, dt)
            blob = [blob for blob in blobs if dl_dt.strftime("%Y%j%H%M%S") in blob.name][0]
            objectId = blob.path.replace('%2F', '/').replace('/b/{}/o/'.format(GOES_PUBLIC_BUCKET), '') #need to be fixed: find the nearest time not first time
            # if dt.year == 2019:
            #     print(1)
            logger.info('Found %s for %s',objectId, str(dt))
            return objectId
        else:

            logger.error(
                'No matching files found for gs://%s/%s* containing %s',GOES_PUBLIC_BUCKET, gcs_prefix,
                                                                            gcs_patterns)
            return None

    def copy_fromgcs(bucket, objectId, destdir):
        """
        Gets GOES object ID from GCS and returns its path

        Args:
            bucket: gcs bucket
            objectId: objectId to request
            destdir: destiny directory

        Returns:
            file path to nc file
        """
        logger = get_logger(LOG_FILE)
        storage_client = gcs.Client.create_anonymous_client()
        bucket = storage_client.get_bucket(bucket)
        blob = bucket.blob(objectId)
        basename = os.path.basename(objectId)

        logger.info('Downloading %s', basename)
        dest = Path(destdir, basename)
        Path(dest).parent.mkdir(parents=True, exist_ok=True)

        # if not Path.is_file(dest):
        blob.download_to_filename(dest)
        return dest

    GOES_PUBLIC_BUCKET = 'gcp-public-data-goes-16'

    def list_gcs(bucket_name, gcs_prefix, gcs_patterns):
        """
        Lists available blobs that match prefix and patterns

        Args:
            bucket_name: gcs bucket name
            gcs_prefix: gcs directory prefix to match
            gcs_patterns: gcs file pattern to match

        Returns:
            list of blobs that match prefix and pattern
        """
        storage_client = gcs.Client.create_anonymous_client()
        bucket = storage_client.get_bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=gcs_prefix, delimiter='/')
        result = []
        if gcs_patterns == None or len(gcs_patterns) == 0:
            for b in blobs:
                result.append(b)
        else:
            for b in blobs:
                match = True
                for pattern in gcs_patterns:
                    if not pattern in b.path:
                        match = False
                if match:
                    result.append(b)
        return result

    
    def _get_goes_time(curr_dt, product, name, channel):
        success = False
        logger = get_logger(LOG_FILE)
        ## Try two times because sometime it fails reading the file

        try:
            ## Get object_id in google cloud services
            object_id = get_object_id_at(curr_dt, products=product, channel=channel)
            ## Download file (in NetCDF format) and store location
            dest_dir = raw_dir/str(curr_dt.year)/curr_dt.strftime('%m')/curr_dt.strftime('%d')/curr_dt.strftime('%H')
            local_file_name = copy_fromgcs('gcp-public-data-goes-16', objectId=object_id, destdir=dest_dir)
            ## Get dataset from file

            tif_file = local_file_name.with_suffix('.tif')
            tif_file = Path(str(tif_file).replace('raw_nc','raw'))
            tif_file.parent.mkdir(parents = True, exist_ok=True)
            if (not Path.is_file(tif_file)):
                ## Save file as tiff
                with rioxarray.open_rasterio(local_file_name) as s:
                    # tif_file = os.path.join(out_dir,'files/'+Path(str(local_file_name)).stem + "_" + product + '.tif') 
                    s[name].rio.to_raster(tif_file)
            else:
                logger.info(f'\nFound file {tif_file} on disk')
            success = True        
        except:
            success = False
        return success
    logger = get_logger(LOG_FILE)
    
    for delta_t in range(-60, 1, 10):
        download_dt = dt + timedelta(minutes=delta_t)
        for product, value in product_dict.items():
            name = value['name']
            channels = value['channels']
            for channel in channels:                
                success = _get_goes_time(download_dt, product, name, channel)        
                      
 
        
    gc.collect()
    return success

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


def main():
    """_summary_
    """
    raw_dir = DATA_DIR/'goes'/'raw_nc'

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
    
    config = parser.parse_args()
    config = vars(config)

    # dt = datetime.now() - timedelta(minutes = 60)
    
    dt = datetime(int(config['year']), int(config['month']), int(config['day']), int(config['hour']), 0, 0, 0)
    
    # TODO Add argparse debug
    debug = True
    if debug:
        # for hour in range(1,7):
        # dt = datetime(2020,7,7,21)
        dt = datetime(2020,6,3,21)
        dt = datetime(2020,6,4,11)
            # dt = dt + timedelta(hours = hour)

        config_file = CONFIG_DIR/'config_goes.json'

        with open(config_file) as f:
            products = json.load(f)
        print(dt)
        get_goes(dt=dt, raw_dir=raw_dir, product_dict = products)
    else:
        config_file = CONFIG_DIR/'config_goes.json'

        with open(config_file) as f:
            products = json.load(f)
        print(dt)
        get_goes(dt=dt, raw_dir=raw_dir, product_dict = products)
    


    # # 3 bands
    # products = {'ABI-L2-CMIPC':{'name':'CMI', 'channels':[f'C{i:02d}' for i in range(1,2)]},
    #             'ABI-L2-ACHTF':{'name':'TEMP', 'channels':['']}, 
    #             'ABI-L2-ACTPC':{'name':'Phase', 'channels':['']}
    # }
    # get_goes(dt=dt, raw_dir=raw_dir, product_dict = products)

    print('end')

if __name__ == '__main__':
    main()
