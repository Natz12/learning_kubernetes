import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from logging import handlers
from pathlib import Path

import mgrs
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# https://pytorch.org/tutorials/recipes/recipes/save_load_across_devices.html

sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))
from model_utils import create_loader, get_prediction, save_results

from C_2_stack import stack_arrays
from model.cnn import UnetSegmentation

DATA_DIR = Path(__file__).resolve().parent.parent/'data'

LOG_DIR = DATA_DIR/"logs"

LOG_FILE = True

logging.root.handlers = []


def get_logger(log_file):
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
            fh = handlers.RotatingFileHandler(LOG_DIR / "goes_predict.log", maxBytes=(1048576 * 5), backupCount=7)
            fh.setFormatter(log_format)
            log.addHandler(fh)
    return log


logger = get_logger(LOG_FILE)



def main():

    latitude = 46.511909
    longitude = -109.156326

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

    # dt = datetime.now() - timedelta(minutes = 60)
    
    dt_center = datetime(int(config['year']), int(config['month']), 
            int(config['day']), int(config['hour']), 0, 0, 0)
    
    debug = True
    # TODO Add argparse debug
    if debug:
        latitude = float(config['latitude'])
        longitude = float(config['longitude'])
        # for hour in range(30):

        dt_center = datetime.now()
        dt_center = dt_center.replace(year = 2020)
        dt_center = datetime(2020,6,4,11)
        # dt_center = datetime(2020,5,18,18)
        # dt_center = dt_center + timedelta(hours = hour)

        dt_center = dt_center - timedelta(hours = 2)

        # Create lodader
        array_dir = DATA_DIR/'goes/np_arrays'

        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
        area = f'area_{mgrs_coordinates}'

        array_dir = DATA_DIR/Path('goes/np_arrays')
        array_dir = array_dir/area

        x = stack_arrays(dt_center,array_dir)
        loader = create_loader(x)
        # Load
        model_path = DATA_DIR/'models/goes/event_no_event/Fold_1_classify_3.pth.tar'
        device = torch.device('cpu')
        model = UnetSegmentation()
        trained_model = torch.load(model_path, map_location=device)
        model.load_state_dict(trained_model['state_dict'])

        # predict
        prediction, scores = get_prediction(model,loader,device)

        # Save prediction    
        area_dir = DATA_DIR/f'predictions/{dt_center.year}/goes'
        area_dir.mkdir( parents = True, exist_ok=True)
        area_file = area_dir/f'{dt_center.year}_area_{latitude:.4f}_{longitude:.4f}.csv'
        print(f'saving prediction with dt_center: {dt_center}, {prediction.item()}')
        save_results(prediction.item(), scores, dt_center, area_file)
    else:
        latitude = float(config['latitude'])
        longitude = float(config['longitude'])

        dt_center = dt_center - timedelta(hours = 2)

        # Create lodader
        array_dir = DATA_DIR/'goes/np_arrays'

        m = mgrs.MGRS()
        mgrs_coordinates = m.toMGRS(latitude, longitude, MGRSPrecision = 2)
        area = f'area_{mgrs_coordinates}'

        array_dir = DATA_DIR/Path('goes/np_arrays')
        array_dir = array_dir/area

        x = stack_arrays(dt_center,array_dir)
        loader = create_loader(x)
        # Load
        model_path = DATA_DIR/'models/goes/event_no_event/Fold_1_classify_3.pth.tar'
        device = torch.device('cpu')
        model = UnetSegmentation()
        trained_model = torch.load(model_path, map_location=device)
        model.load_state_dict(trained_model['state_dict'])

        # predict
        prediction, scores = get_prediction(model,loader,device)

        # Save prediction    
        area_dir = DATA_DIR/f'predictions/{dt_center.year}/goes'
        area_dir.mkdir( parents = True, exist_ok=True)
        area_file = area_dir/f'{dt_center.year}_area_{latitude:.4f}_{longitude:.4f}.csv'
        print(f'saving prediction with dt_center: {dt_center}, {prediction.item()}')
        save_results(prediction.item(),scores, dt_center, area_file)
    print('end')
    return prediction.item()

if __name__ == '__main__':
    main()


