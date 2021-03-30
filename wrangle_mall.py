import pandas as pd
import numpy as np
import os
from env import host, user, password

# Establish a connection
def get_connection(db, user=user, host=host, password=password):
    '''
    This function uses my info from my env file to
    create a connection url to access the CodeUp db.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def new_mall_data():
    '''
    This function reads the mall_customers data from the CodeUp db into a df.
    '''
    sql_query = '''
                select *
                from customers;
                '''
    df = pd.read_sql(sql_query, get_connection('mall_customers'))

    return df

def get_mall_data(cached=False):
    '''
    This function reads in  mall_customers data from Codeup database and returns it
    as a .csv file containing a single dataframe. 
    '''
    
    filename = "mall.csv"
    if cached == False or os.path.isfile(filename) == False:
        df = new_mall_data()
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename, index_col=0)
      
   
    return df

def get_object_cols(df):
    '''
    This function takes in a dataframe and identifies the columns that are object types
    and returns a list of those column names. 
    '''
    # create a mask of columns whether they are object type or not
    mask = np.array(df.dtypes == "object")

        
    # get a list of the column names that are objects (from the mask)
    object_cols = df.iloc[:, mask].columns.tolist()
    
    return object_cols

def create_dummies(df, object_cols):
    '''
    This function takes in a dataframe and list of object column names,
    and creates dummy variables of each of those columns. 
    It then appends the dummy variables to the original dataframe. 
    It returns the original df with the appended dummy variables. 
    '''
    
    # run pd.get_dummies() to create dummy vars for the object columns. 
    # we will drop the column representing the first unique value of each variable
    # we will opt to not create na columns for each variable with missing values 
    # (all missing values have been removed.)
    dummy_df = pd.get_dummies(df[object_cols], dummy_na=False, drop_first=True)
    
    # concatenate the dataframe with dummies to our original dataframe
    # via column (axis=1)
    df = pd.concat([df, dummy_df], axis=1)

    return df

def train_validate_test(df, target):
    train, test = train_test_split(mall_df, train_size = 0.8, random_state = 123)
    train, validate = train_test_split(train, train_size = 0.75, random_state = 123)

    return train, test, validate