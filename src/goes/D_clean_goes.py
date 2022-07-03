import argparse
import logging
import os
import shutil
import time
from datetime import datetime
from logging import handlers
from pathlib import Path
import subprocess
from typing import Union
import sys

sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))

from misc_utils import get_date_from_name


# DATA_DIR = Path(__file__).resolve().parent.parent/'data'
DATA_DIR = Path(__file__).resolve().parent.parent/'data'

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
            fh = handlers.RotatingFileHandler(LOG_DIR / "goes_clean.log", maxBytes=(1048576 * 5), backupCount=7)
            fh.setFormatter(log_format)
            log.addHandler(fh)
    return log


logger = get_logger(LOG_FILE)

def mv_to_s3(directory_to_move:Union[Path,str], 
            s3_directory:str,
            timeout:int = 14400):

    # directory_to_move = next(data_dir.rglob('*.h5')).parent
    aws = os.popen('which aws').read().strip()
    
    try:
        subprocess.run([aws, 
                        's3', 
                        'mv',
                        directory_to_move,
                        s3_directory,
                        '--recursive', 
                        '--profile',
                        'radar'
                        ],
                        timeout = timeout,
                        check = True
                    )
    except subprocess.TimeoutExpired:
        logger.warning(f'Timeout expired for moving files from '+\
            '{directory_to_move} to S3 bucket {s3_directory}')
        # logger.warning('Timeout expired for downloading HRDPS')
    except subprocess.CalledProcessError as e:
        logger.error(f'Error moving files from {directory_to_move} to'+\
            ' S3 bucket {s3_directory}\n {e}')
        
    return s3_directory

# def clean(temp_dst_directory,
#             zip_dir):
#     try:        
#         for root, _, _ in os.walk(temp_dst_directory, topdown=False):
#             os.rmdir(root)
#         os.remove(zip_dir)
#     except:
#         print(f'Directory {root} was not empty')

def mv_clean(base_dir:Union[Path,str],
        num_days:int=30, 
        s3_base:str='s3://airm-data-01/shared-data/Hail/0_dataset/data/GOES'):
    """ moves files to S3 bucket and deletes empty folders
    data will be moved to an S3 bucket with name s3_base/<year of file>/s3_folder
    where s3_folder will be 'raw', if base_dir contains the word 'raw', or 'array' otherwise  

    Args:
        base_dir (Union[Path,str]): base directory walk looking for files
        num_days (int, optional): number of days that have to pass before moving files and erasing folders. Defaults to 30.
        s3_base (str, optional): S3 base directory. Defaults to 's3://airm-data-01/shared-data/Hail/0_dataset/data/GOES/'.

    Returns:
        str: destination s3 bucket
    """
    
    # try:        
    #     for root, _, _ in os.walk(base_dir, topdown=False):
    #         print("root: ",root,"\ncomparison:", str(comparison_dir),
    #             'root<str(comparison_dir)',root<str(comparison_dir))
    #         # os.rmdir(root) # Removes only if it is empty
    #     # os.remove(zip_dir) # Removes even if it is not empty
    # except:
    #     print(f'Directory {root} was not empty')
    num_days = 86400*num_days
    now = time.time()
    s3_directory = None
    for r,d,f in os.walk(base_dir, topdown=False):
        
        if f:
            timestamp = os.path.getmtime(r)
            if now-num_days > timestamp:
                s3_folder = 'raw' if 'raw' in str(r) else 'array'
                dt = get_date_from_name(Path(f[0]),'s')
                s3_directory = os.path.join(s3_base, str(dt.year), s3_folder)
                mv_to_s3(os.path.join(r), 
                        s3_directory,
                        timeout= 14400)
                os.rmdir(r)
        for dir in d:
            timestamp = os.path.getmtime(os.path.join(r,dir))
            if now-num_days > timestamp:
                try:
                    os.rmdir(r)
                    # shutil.rmtree(os.path.join(r,dir))  #uncomment to use
                    print("removed ",os.path.join(r,dir))
                except Exception as e:
                    print(e)
                    pass
                else: 
                    print("some message for success")
    return s3_directory

def clean(base_dir, num_days=30):

    num_days = 86400*num_days
    now = time.time()

    for r,d,f in os.walk(base_dir, topdown=False):
        for dir in d:
            timestamp = os.path.getmtime(os.path.join(r,dir))
            if now-num_days > timestamp:
                try:
                    print("removing ",os.path.join(r,dir))
                    shutil.rmtree(os.path.join(r,dir))  
                except Exception as e:
                    print(e)
                    pass

def main():
    """_summary_
    """

    raw_dir = DATA_DIR/'goes'/'raw'
    array_dir = DATA_DIR/'goes'/'np_arrays'

    parser = argparse.ArgumentParser()
    parser.add_argument('-yr','--year',
                    default=str(2020),
                    help='4-digit year to be downloaded')
    parser.add_argument('-m','--month',
                    default=str(1),
                    help='1 or 2 digit month to be downloaded')
    parser.add_argument('-d','--day',
                    default=str(1),
                    help='1 or 2 digit day to be downloaded')
    parser.add_argument('-hr','--hour',
                    default=str(0),
                    help='1 or 2 digit hour. Hour-1 will be downloaded')
    
    config = parser.parse_args()
    config = vars(config)
    dt = datetime(int(config['year']), int(config['month']), int(config['day']), int(config['hour']), 0, 0, 0)

    dt_dir = Path(str(dt.year))/dt.strftime('%m')/dt.strftime('%d')/dt.strftime('%H')
    

    # folder = ""
    mv_clean(raw_dir,30)
    clean(array_dir,30)
    print('end')


if __name__ == "__main__":
    main()
