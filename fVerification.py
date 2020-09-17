import numpy as np
import pandas as pd
from config import *


class fVerification:
    
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
        
        necessary_cols = D_NECESSARY_COLS
                  
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

        necessary_colTypes = D_NECESSARY_COLTYPES
            
        for key in necessary_colTypes.keys():
            if col_Types[key] == necessary_colTypes[key]:
                True
            else:
                raise TypeError(COLUMN_TYPE_ERROR.format(key,col_Types[key],necessary_colTypes[key]))
        return True
    
    
    def run(self):
        
        if self.NotEmpty():
            print("fVerification|Not Empty DataFrame     : +")
        if self.ColumnAvailaibility():
            print("fVerification|Column availaibility    : +")
        if self.ColumnType():
            print("fVerification|Column Type correctness : +")
            
        return True