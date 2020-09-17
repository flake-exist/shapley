import numpy as np
import pandas as pd

#---COLUMN NAMES CONSTANTS
USER_PATH     = "user_path"
COUNT         = "count"
CHANNEL_NAME  = "channel_name"
CHANNEL_SEQ   = "channel_seq"
SHAPLEY_VALUE = "shapley_value"
POSITION      = "position"
TIMELINE      = "timeline"
CLIENT_ID     = "ClientID"
DATE_START    = "date_start"
DATE_FINISH   = "date_finish"
STATUS        = "status"

PRECISION = 5 # Digits after point
ERROR = 0.01


#---VERIFICATION CONSTANTS
NECESSARY_COLS = [USER_PATH,COUNT]
NECESSARY_COLTYPES = {USER_PATH:np.object,COUNT:np.int}

D_NECESSARY_COLS = [CLIENT_ID,USER_PATH,TIMELINE]
D_NECESSARY_COLTYPES = {CLIENT_ID:np.object,USER_PATH:np.object,TIMELINE:np.object}

CHANNEL_DELIMITER  = "=>"
TIMELINE_DELIMITER = "=>"

DAY_PATTERN = r'^\d{1,10}[Dd]$'
MONTH_PATTERN = r'^\d{1,10}MS$'
THOUSAND=1000
TIME_ZONE = 3

#---FUNCTIONS---

# Limit huge chains by quantile value
# def ChainLimit(data,sep,quantile=0.995):
    
#     init_size = data.shape[0]
#     init_count_sum = data[COUNT].sum()
#     init_columns = list(data.columns)
    
#     data['size'] = data[USER_PATH].apply(lambda x: len(x.split(sep)))
#     thresh_val = data['size'].quantile(quantile)
    
#     data_edit = data[data['size'] >  thresh_val][init_columns]
#     data_save = data[data['size'] <= thresh_val][init_columns]
    
#     data_edit[USER_PATH] = data_edit[USER_PATH].apply(lambda x: sep.join(x.split(sep)[:int(thresh_val)]))
    
#     data_new = pd.concat([data_save,data_edit])
    
#     if (data_new.shape[0] == init_size) & (data_new[COUNT].sum() == init_count_sum):
#         return data_new
#     else:
#         raise ValueError('''Number of chains or sum of conversion in new DataFrame does not equal init number of chains or sum of conversionins old DataFrame''')

def ChainSplit(chain,channel_delimiter): return chain.split(channel_delimiter)

def GetEncoding(sequence,unique_channels=True):
    '''
    Create `self.encryped_dict` indicating each unique channel its unique ID
    Format -> {channel_1 : id_1,
                 ...
                 ...
                 ...
               channel_n : id_n}

    INPUT  : [1] `sequence`         - list with unique channels
    RETURN : [1] `channel_id_dict`  - encoded unique channels dict
    '''
    channel_id_dict = {}
    
    if unique_channels == True:
        sequence_unique = sequence
    else:
        sequence_unique = set(sequence)

    for id_,channel in enumerate(sequence_unique):
        channel_id_dict[channel] = id_

    return channel_id_dict

def SequenceEncode(sequence_toEncode,channel_id_dict,chain_size_limit): 
    '''
    Convert channel[s] into its id[s]
    INPUT  : [1] `sequence_toEncode` - list of channels
             [2] `channel_id_dict`   - encoded unique channels dict
    RETURN : [1] array of channel ids (Array[Int])
        '''
    return np.array([channel_id_dict[channel] for channel in sequence_toEncode])[:chain_size_limit]

def Cardinality(M_buffer):
    '''
    Cardinality calculation
    INPUT  : [1] `M_buffer` - masked M
    RETURN : [1] array of path cardinalities (Array[Int])
    '''
    cardinality = []
    for i in range(M_buffer.shape[0]):
        row = M_buffer[i,:]
        cardinality.append(np.unique(row[~np.isnan(row)]).shape[0]) #calculate Cardinality (non empty values)
    return np.array(cardinality)

def DecodeDict(encoded_dict,channel_id_dict):
    '''
    Decode `encoded_dict`, where each unique channel is encoded into unique id
    Before: -> {0:val1,
                1:val2,
                ...,
                ...,
                ...,
                N: val_n} wherenN - number of unique channels and val_n its value

    After-> {channel_name_1:val1,
             channel_name_2:val2,
                ...,
                ...,
                ...,
             channel_name_n: val_n} where n - number of unique channels and val_n its value
    INPUT  : [1] `encoded_dict`       - encoded_dict (shapley_classic or shapley_order dict (Shapley algorithm result)
             [2] `channel_id_dict`    - encoded unique channels dict
    RETURN : [1] `decoded_dict`
    '''
    decoded_dict = {}

    invert_channel_dict = {v: k for k, v in channel_id_dict.items()} 

    for key in encoded_dict.keys():
        decoded_dict[invert_channel_dict[key]] = encoded_dict[key]

    return decoded_dict

def ShapleyOrderToFrame(shapley_order,columns=[CHANNEL_NAME,POSITION,SHAPLEY_VALUE]):
    '''
    Convert nested `shapley_order` dict into DataFrame[`CHANNEL_NAME`,`POSITION`,`SHAPLEY_VALUE`]
    INPUT  : [1] `shapley_order`- shapley_order dict (Shapley algorithm result)
    RETURN : [1]  DataFrame
    '''
    row_list = []
    for key_channel in shapley_order.keys():
        for key_position in list(shapley_order[key_channel].keys()):
            row = (key_channel,key_position,shapley_order[key_channel][key_position])
            row_list.append(row)
    data = pd.DataFrame(row_list,columns=columns)
    return data

def periodsCombinator(seq):
    index = 0
    store = []
    while index < len(seq) - 1:
        row = (seq[index],seq[index+1])
        store.append(row)
        index += 1
    return store

#---ERROR MESSAGES---

SHAPLEY_ORDER_ERROR        = "Required : Sum of {0} touchpoint values - {1} | Found : Sum of {0} touchpoint values - {2}"
EMPTY_DATAFRAME_ERROR      = "DataFrame is Empty"
COLUMN_AVAILAIBILITY_EEROR = "Find columns: {0}\n Required columns : {1}"
COLUMN_TYPE_ERROR          = "Find : {0}[{1}]\nRequired : {0}[{2}]"
UNIQUE_CHAIN_ERROR         = "`{0}` must contain only unique (aggregated) paths"
INTERVAL_CREATOR_ERROR     = "Incorrect frequency format.You can use days and months for exploring periods.Example : For days - 1D or 14d, For months - 1MS or 2MS"
MILISECONDS_FORMAT_ERROR   = "Milliseconds format has Boolean type only"
EFFICIENCY_ERROR           = "Total campaign contribution equals the sum of Shapley values contributions.Find : total_campaign - {0} | Shapley values sum : {1}"
DUMMY_PLAYER_ERROR         = "Sum of Dummy Players equals 0. Channels did not contribute any value could not have Shaple value greater than 0.Find : Sum of Dummy Players : {}"
CONVERGE_ERROR             = "`shapley_order` values does not converge `shapley_classic` values"
