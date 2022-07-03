import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import re
import sys
import os
import pandas as pd

sys.path.insert(1, os.path.join(sys.path[0],'..','utils'))
from misc_utils import select_files_dt,dates_list_from_base, get_goes_date,get_date_from_name


def stack_arrays(dt_center,
        array_dir,
        product_dic= None, 
        minutes_frequency = 10):
    """_summary_

    Args:
        dt_center (_type_): _description_
        array_dir (_type_): _description_
        product_dic (_type_): _description_
    """
    file_list = []
    for delta_t in range(-60, 61, 60):
        dt = dt_center + timedelta(minutes = delta_t)
        dt_dir = Path(str(dt.year))/dt.strftime('%m')/dt.strftime('%d')/dt.strftime('%H')
        array_dt_dir = array_dir/dt_dir
        file_list.append(sorted(array_dt_dir.rglob('*.npy')))

    file_list = [item for sublist in file_list for item in sublist]
    
    files_df = select_files_dt(file_list, dt_center, minutes_frequency)

    # ['a', 'b'], ascending=[True, False]

    time_arrays = []
    for unique_dt in files_df['date'].unique():
        files_dt_df = files_df[files_df['date'] == unique_dt]
        product_arrays = []
        for _, row in files_dt_df.iterrows():
            product_arrays.append(np.load(row['file']))
        product_stack = np.stack(product_arrays)
        time_arrays.append(product_stack)
        time_stack = np.stack(time_arrays)
    time_stack=time_stack.reshape(-1, time_stack.shape[2], time_stack.shape[3])

    return time_stack


if __name__ == '__main__':
    # array_dir = Path('/media/hail/git_naty/hail_monitoring/data/goes/np_arrays')
    # dt_center = datetime(2022, 5, 7, 1, 0, 0, 0)
    # product_dic =  {'ABI-L2-CMIPC':{'name':'CMI', 'channels':[f'C{i:02d}' for i in range(1,17)]},
    #     'ABI-L2-ACHTF':{'name':'TEMP', 'channels':['']}, 
    #     'ABI-L2-ACTPC':{'name':'Phase', 'channels':['']}
    # }  
    # stack_arrays(dt_center,array_dir,product_dic)
    print('end')
