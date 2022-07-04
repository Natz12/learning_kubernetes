from datetime import datetime, timedelta
import re
from pathlib import Path
import pandas as pd

def dates_list_from_base(start, stop, step):
    dates_list = [start + timedelta(minutes = minutes) for minutes in range(0,stop,step)]
    return dates_list

def nearest_date(dates, pivot):
    return min(dates, key=lambda x: abs(x - pivot))

def get_goes_date(file_name, letter):
    dt = datetime.strptime(re.search('%s[0-9]{11}' % letter, file_name.stem)[0].replace('%s' % letter,''),'%Y%j%H%M')
    return dt

def get_date_from_name(file_name, letter, num_occurrences = 11, pattern = '%Y%j%H%M'):
    dt = datetime.strptime(re.search('%s[0-9]{%i}' % (letter, num_occurrences), file_name.stem)[0].replace('%s' % letter,''),pattern)
    return dt

def get_goes_start_date(file_name):
    dt = datetime.strptime(re.search('s[0-9]{11}', file_name.stem)[0].replace('s',''),'%Y%j%H%M')
    return dt

def get_goes_monitoring_date(file_name):
    dt = datetime.strptime(re.search('m[0-9]{11}', file_name.stem)[0].replace('m',''),'%Y%j%H%M')
    return dt

def get_dates_from_files(file_list):

    dates_file_list = [get_date_from_name(f, 'm') for f in file_list]
    
    return dates_file_list

def select_files_dt(file_list, dt_center, minutes_frequency):
    file_stem_list = [Path(item).stem for item in file_list]
    dates_file_list = get_dates_from_files(file_list)

    minutes_add = 121

    dates_monitoring_list = dates_list_from_base(dt_center-timedelta(minutes = 60), minutes_add, minutes_frequency)

    file_dic = {'file':file_list, 'date': dates_file_list, 'file_stem': file_stem_list}
    files_df = pd.DataFrame(file_dic)
    # files_df = pd.DataFrame(list(zip(file_list,dates_file_list)),
    #            columns =['file', 'date'])

    files_df = files_df[files_df['date'].isin(dates_monitoring_list)]
    files_df = files_df.sort_values(by = ['date','file'])
    return files_df

