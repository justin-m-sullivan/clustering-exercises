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

def new_zillow_data():
    '''
    This function reads the Zillow data from the CodeUp db into a df.
    '''
    sql_query = '''
          SELECT prop.*, 
                pred.logerror, 
                pred.transactiondate, 
                air.airconditioningdesc, 
                arch.architecturalstyledesc, 
                build.buildingclassdesc, 
                heat.heatingorsystemdesc, 
                landuse.propertylandusedesc, 
                story.storydesc, 
                construct.typeconstructiondesc 

            FROM   properties_2017 prop  
            INNER JOIN (SELECT parcelid,
       					  logerror,
                          Max(transactiondate) transactiondate 
                        FROM   predictions_2017 
                        GROUP  BY parcelid, logerror) pred
                    USING (parcelid) 
                LEFT JOIN airconditioningtype air USING (airconditioningtypeid) 
                LEFT JOIN architecturalstyletype arch USING (architecturalstyletypeid) 
                LEFT JOIN buildingclasstype build USING (buildingclasstypeid) 
                LEFT JOIN heatingorsystemtype heat USING (heatingorsystemtypeid) 
                LEFT JOIN propertylandusetype landuse USING (propertylandusetypeid) 
                LEFT JOIN storytype story USING (storytypeid) 
                LEFT JOIN typeconstructiontype construct USING (typeconstructiontypeid) 
                    WHERE  prop.latitude IS NOT NULL 
                        AND prop.longitude IS NOT NULL
                '''
    df = pd.read_sql(sql_query, get_connection('zillow'))

    return df


# Acquire Data
def get_zillow_data(cached=False):
    '''
    This function reads in zillow data from Codeup database and returns it
    as a .csv file containing a single dataframe. 
    '''
    
    filename = "zillow.csv"
    if cached == False or os.path.isfile(filename) == False:
        df = new_zillow_data()
        df.to_csv(filename)
    else:
        df = pd.read_csv(filename, index_col=0)
      
   
    return df

#Summarize Data 

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

def get_numeric_cols(df, object_cols):
    '''
    takes in a dataframe and list of object column names
    and returns a list of all other columns names, the non-objects. 
    '''
    numeric_cols = [col for col in df.columns.values if col not in object_cols]
    
    return numeric_cols

def get_single_use_prop(df):
    single_use = [261, 262, 263, 264, 266, 268, 273, 276, 279]
    df = df[df.propertylandusetypeid.isin(single_use)]
    return df 

def handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5):
    ''' funtcion which takes in a dataframe, required notnull proportions of non-null rows and columns.
    drop the columns and rows columns based on theshold:'''
    
    #drop columns with nulls
    threshold = int(prop_required_col * len(df.index)) # Require that many non-NA values.
    df.dropna(axis = 1, thresh = threshold, inplace = True)
    
    #drop rows with nulls
    threshold = int(prop_required_row * len(df.columns)) # Require that many non-NA values.
    df.dropna(axis = 0, thresh = threshold, inplace = True)
    
    
    return df

def clean_zillow(df):
    df = get_single_use_prop(df)

    df = handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5)

    cols_to_drop = ['fullbathcnt','heatingorsystemtypeid','finishedsquarefeet12', 
                'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 'censustractandblock',
                'propertylandusedesc', 'buildingqualitytypeid' , 'unitcnt', 'heatingorsystemdesc', 
                'lotsizesquarefeet','regionidcity', 'calculatedbathnbr']

    df.drop(columns=cols_to_drop, inplace = True)

    df.dropna(inplace = True)

    df.set_index('parcelid', inplace=True)

    df.drop(columns={'id'}, inplace=True)

    return df