import pandas as pd
from config import *

def FilterTheDict(dictObj,callback):
    dummyDict = dict()
    for (key, value) in dictObj.items():
        if callback((key, value)):
            dummyDict[key] = value
    return dummyDict

class Properties:
    
    def __init__(self,data,shapley_classic,path_col=USER_PATH,count_col=COUNT):
        self.data            = data
        self.shapley_classic = shapley_classic
        self.path_col        = path_col
        self.count_col       = count_col
    
    
    
    def Efficiency(self):
        '''
        This ensures that sum of Shapley values (attribution values) of channels equals to the total
        campaign value
        '''
        
        shapley_val_sum = round(float(sum(self.shapley_classic.values())),1)
        total_campaign  = round(float(self.data[self.count_col].sum()),1)
        
        if shapley_val_sum == total_campaign:
            return True
        else:
            raise ValueError(EFFICIENCY_ERROR.format(total_campaign,shapley_val_sum))
            
    def run(self):
        self.Efficiency()
        
#     def DummyPlayer(self,M):
#         '''
#         The channel that makes no contribution to any coalition will get zero credit
#         '''
#         dummyDict = FilterTheDict(self.shapley_classic, lambda elem : elem[1] == 0) # IDs(channels) having zero credit (value)
#         M_dummy = M[:,list(dummyDict.keys())] # Matrix consists of IDs(channels) with zero credit (value)
#         dummy_ID = np.unique(np.where(M_dummy > 0)[0]) 
#         dummy_sum = self.data[self.count_col].values[dummy_ID].sum()
        
#         if dummy_sum == 0:
#             return True
#         else:
#             raise ValueError(DUMMY_PLAYER_ERROR.format(dummy_sum))
