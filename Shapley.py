import pandas as pd
import numpy as np
import argparse
from Verification import Verification
from Properties import Properties
from config import *

#Shapley'

class Shapley:
    
    def __init__(self,data,chain_size_limit=150):
        
        self.data              = data
        self.channel_id_dict   = None
        self.chain_size_limit  = chain_size_limit
        self.conversions = self.data[COUNT].values
        
    def PathStats(self):
        
        path_count = self.data[USER_PATH].shape[0] #path_count

        self.data[CHANNEL_SEQ] = self.data[USER_PATH].apply(lambda x:ChainSplit(x,CHANNEL_DELIMITER)[0:self.chain_size_limit])
        channel_list             = self.data[CHANNEL_SEQ].values

        flat_channel_list = [channel for sublist in channel_list for channel in sublist]
        path_length_max   = max([len(sublist) for sublist in channel_list]) #path_length_max

        unique_channel_list  = set(flat_channel_list)
        self.channel_id_dict = GetEncoding(unique_channel_list)
        unique_channel_count = len(unique_channel_list) #unique_channel_count
        
        return path_count,unique_channel_count,path_length_max
    
    def Vectorization(self,path_count,unique_channel_count,path_length_max):
        
        M = np.empty((path_count,self.chain_size_limit))
        M.fill(np.nan)
        
        for index,path in enumerate(self.data[CHANNEL_SEQ]):
            path_encoded = SequenceEncode(path,self.channel_id_dict,chain_size_limit=self.chain_size_limit)
            M[index,0:path_encoded.shape[0]] = path_encoded
        
        return M
    
    def Classic(self,M):
        
        row_storage = []
        
        for row,conversion in zip(M,self.conversions):
            clean_row = np.unique(row[~np.isnan(row)])
            row_storage.append((clean_row,np.round(conversion/clean_row.shape[0],10)))
            
        row_data = pd.DataFrame(row_storage,columns=[CHANNEL_NAME,SHAPLEY_VALUE])
        row_agg = row_data.explode(CHANNEL_NAME).groupby([CHANNEL_NAME])[SHAPLEY_VALUE].sum()
        row_agg.index = row_agg.index.astype(int)
        
        shapley_Encoded = row_agg.to_dict()
        
        shapley_classic = DecodeDict(shapley_Encoded,self.channel_id_dict)
        
        Properties(self.data,shapley_classic).run() #Check Properties
        
        shapley_classic = pd.DataFrame(shapley_classic.items(),columns=[CHANNEL_NAME,SHAPLEY_VALUE])
         
        return shapley_classic

    def Order(self,M):

        rowpos_storage = []
        
        for row,conversion in zip(M,self.conversions):
            clean_row = row[~np.isnan(row)]
            unique, counts = np.unique(clean_row, return_counts=True)
            id_count_dict  = dict(zip(unique, counts))
            intermediate_value = conversion / unique.shape[0]
            value_set = [np.round(intermediate_value / id_count_dict[id_],10) for id_ in clean_row]
            rowpos_storage.append((clean_row,value_set,np.arange(clean_row.shape[0])))

        data = pd.DataFrame(rowpos_storage,columns=[CHANNEL_NAME,SHAPLEY_VALUE,POSITION])

        channel_col  = data[CHANNEL_NAME].explode()
        value_col    = data[SHAPLEY_VALUE].explode()
        position_col = data[POSITION].explode()

        pos_data = pd.concat([channel_col,value_col,position_col],axis=1)

        pos_agg = pos_data.groupby([CHANNEL_NAME,POSITION])[SHAPLEY_VALUE].sum()

        pos_agg = pos_agg.reset_index([CHANNEL_NAME,POSITION])
        
        pos_agg[CHANNEL_NAME] = pos_agg[CHANNEL_NAME].astype(int)
        
        inv_channel_id_dict = {v: k for k, v in self.channel_id_dict.items()} #inverted dict
        
        pos_agg[CHANNEL_NAME] = pos_agg[CHANNEL_NAME].map(inv_channel_id_dict)
        
        return pos_agg
        
            
    
    def run(self,date_start=None,date_finish=None):
        
        Verification(self.data).run() # Verification
        print("Data Verificationed  : True")
        
        path_count,unique_channel_count,path_length_max = self.PathStats()
        print("Stats Verificationed : True")
        
        M = self.Vectorization(path_count,unique_channel_count,path_length_max)
        print("Matrix Vectorization : True")
        
        shapley_classic = self.Classic(M)
        shapley_classic[DATE_START],shapley_classic[DATE_FINISH]  = [date_start,date_finish]
        print("Shapley classic calculated : True")
        
        
        shapley_order   = self.Order(M)
        shapley_order[DATE_START],shapley_order[DATE_FINISH]= [date_start,date_finish]
        print("Shapley order calculated   : True")
        
        #---Check `shapley_order`'s values equal `shapley_classic`'svalues--- 
        print("Shapley order checking     : True")
        shapley_order_agg = shapley_order.groupby([CHANNEL_NAME])[SHAPLEY_VALUE].sum()
        shapley_order_agg = shapley_order_agg.reset_index(CHANNEL_NAME)
        
        
        check_data = pd.merge(shapley_classic[[CHANNEL_NAME,SHAPLEY_VALUE]],
                              shapley_order_agg[[CHANNEL_NAME,SHAPLEY_VALUE]],
                              on=CHANNEL_NAME,
                              suffixes=['_classic','_order'])

        
        check_data[STATUS] = abs(check_data[SHAPLEY_VALUE + '_classic'] - check_data[SHAPLEY_VALUE + '_order'])
        if check_data[check_data[STATUS] > ERROR].shape[0] == 0:
            print("Shapley order checked      : True")
            return shapley_classic,shapley_order
        else:
            print(check_data[check_data[STATUS] > ERROR])
            raise ValueError(CONVERGE_ERROR)
            
    
if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('--input_filepath', action='store', type=str, required=True)                                
    my_parser.add_argument('--output_filepath', action='store', type=str, required=True)
    my_parser.add_argument('--output_filepath_order', action='store', type=str, required=True)
    args = my_parser.parse_args()
    
    data = pd.read_csv(args.input_filepath)
  
    
    shapley = Shapley(data)
    shapley_classic_df,shapley_order_df = shapley.run() 
    
    shapley_classic_df.to_csv(args.output_filepath)
    shapley_order_df.to_csv(args.output_filepath_order)