import numpy as np
import pandas as pd
from config import *


class Verification:
    
    '''
    Checking if input DataFrame satisfies further conditions
    '''
    
    def __init__(self,data):
        self.data = data
        
    def NotEmpty(self):
        '''
        Checking if DataFrame is empty or not
        RETURN : [1] Boolean
        '''
        if self.data.shape[0] != 0:
            return True
        else:
            raise ValueError(EMPTY_DATAFRAME_ERROR)
    
    #columnAvailaibility
    def ColumnAvailaibility(self):
        '''
        Checking availaibility all of necessary columns
        RETURN : [1] Boolean
        '''
        cols = list(self.data.columns)
        
        necessary_cols = NECESSARY_COLS
                  
        if all([elem in cols for elem in necessary_cols]):
            return True
        else:
            raise ValueError(COLUMN_AVAILAIBILITY_EEROR.format(cols,necessary_cols))
            
    #columnType
    def ColumnType(self):
        '''
        Checking correctness column types
        RETURN : [1] Boolean
        '''
        col_Types = dict(self.data.dtypes)

        necessary_colTypes = NECESSARY_COLTYPES
            
        for key in necessary_colTypes.keys():
            if col_Types[key] == necessary_colTypes[key]:
                True
            else:
                raise TypeError(COLUMN_TYPE_ERROR.format(key,col_Types[key],necessary_colTypes[key]))
        return True
    
    #uniqueChain
    def UniqueChain(self):
        '''
        Checking if input DataFrame contains only unique paths or not
        RETURN : [1] Boolean
        '''
        path_count = self.data[USER_PATH].shape[0]
        unique_count = self.data[USER_PATH].unique().shape[0]

        if path_count == unique_count:
            return True       
        else:
            raise ValueError(UNIQUE_CHAIN_ERROR.format(USER_PATH))
    
    def run(self):
        
        if self.NotEmpty():
            print("Not Empty DataFrame        : +")
        if self.ColumnAvailaibility():
            print("Column availaibility       : +")
        if self.ColumnType():
            print("Column Type correctness    : +")
        if self.UniqueChain():
            print("Unique Chain availaibility : +")
            
        return True