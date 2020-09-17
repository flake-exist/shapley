import argparse
import pandas as pd
import numpy as np
from fVerification import fVerification
from Shapley import Shapley
from datetime import datetime
from datetime import timedelta
from config import *
import re

class FShapley:
    
    def __init__(self,data,
                 channel_delimiter=CHANNEL_DELIMITER,
                 timeline_delimiter=TIMELINE_DELIMITER,
                 time_zone=TIME_ZONE,
                 milliseconds=True):
        
        self.data = data
        self.channel_delimiter  = channel_delimiter
        self.timeline_delimiter = timeline_delimiter
        self.time_zone = time_zone
        self.milliseconds = milliseconds

    def Prepare(self):

        if self.milliseconds == True:
            divider = THOUSAND
        elif self.milliseconds == False:
            divider = 1
        else:
            raise ValueError(MILISECONDS_FORMAT_ERROR)

        self.data['last_touch'] = self.data['timeline'].apply(lambda x:int(x.split(self.timeline_delimiter)[-1]))
        
        self.data['date_local'] = self.data['last_touch'].apply(lambda x:datetime.fromtimestamp(int(x/divider) + timedelta(hours=TIME_ZONE).seconds))
    
    def intervalCreator(self,date_start,date_finish,freq):

        if re.findall(DAY_PATTERN,freq) != []:
            frequency = freq 
        elif re.findall(MONTH_PATTERN,freq) != []: 
            frequency = freq   
        else:
            raise ValueError(INTERVAL_CREATOR_ERROR)
            
        periods = pd.date_range(start=date_start, end=date_finish,freq=frequency).to_pydatetime()
            
        periodsInterval = periodsCombinator(periods) ####!!!!!!!

        return periodsInterval
            

    def periodData(self,epoch):
        data_epoch = self.data[(self.data['date_local'] >= epoch[0]) &
                               (self.data['date_local'] < epoch[1])]
        data_agg = data_epoch.groupby([USER_PATH])[CLIENT_ID].count()
        data_agg = data_agg.to_frame()
        data_agg.reset_index(level=0, inplace=True)
        data_agg.rename(columns={USER_PATH:USER_PATH,CLIENT_ID:COUNT},inplace=True)

        return data_agg
    
    def run(self,date_start,date_finish,freq):
        
        shapley_classic_list = []
        shapley_order_list  = []
        
        fVerification(self.data).run() #Verification 
        
        self.Prepare() #Date Preparation

        periodsInterval = self.intervalCreator(date_start,date_finish,freq)
        
        for period in periodsInterval:
            data_agg = self.periodData(period)
            
            if data_agg.shape[0] != 0:
                shapley_classic_df,shapley_order_df = Shapley(data_agg).run(date_start=period[0],date_finish=period[1])
                shapley_classic_list.append(shapley_classic_df)
                shapley_order_list.append(shapley_order_df)
            else:
                shapley_classic_list.append(pd.DataFrame())
                shapley_order_list.append(pd.DataFrame())
        
        shapley_classic_df_periods = pd.concat(shapley_classic_list)
        shapley_order_df_periods   = pd.concat(shapley_order_list)
        
        return shapley_classic_df_periods,shapley_order_df_periods
    
if __name__ == '__main__':
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--date_start', action='store', type=str, required=True)
        my_parser.add_argument('--date_finish', action='store', type=str, required=True)
        my_parser.add_argument('--freq', action='store', type=str, required=True)
        my_parser.add_argument('--input_filepath', action='store', type=str, required=True)                                
        my_parser.add_argument('--output_filepath', action='store', type=str, required=True)
        my_parser.add_argument('--output_filepath_order', action='store', type=str, required=True)
        args = my_parser.parse_args()
        
        data = pd.read_csv(args.input_filepath,dtype=object)
        
        shapley_classic_df_periods,shapley_order_df_periods = FShapley(data).run(args.date_start,args.date_finish,args.freq)
        
        shapley_classic_df_periods.to_csv(args.output_filepath)
        shapley_order_df_periods.to_csv(args.output_filepath_order)