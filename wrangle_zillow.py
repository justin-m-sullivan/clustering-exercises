import pandas as pd
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
    sql_query = '''SELECT prop.parcelid, 
                    prop.bathroomcnt, 
                    prop.bedroomcnt, 
                    prop.calculatedbathnbr, 
                    prop.calculatedfinishedsquarefeet, 
                    prop.fips,
                    prop.latitude,
                    prop.longitude,  
                    prop.structuretaxvaluedollarcnt, 
                    prop.taxvaluedollarcnt, 
                    prop.landtaxvaluedollarcnt, 
                    prop.taxamount, 
                    prop.propertylandusetypeid,
                    proptype.propertylandusedesc
            FROM properties_2017 as prop
            JOIN predictions_2017 as pred USING(parcelid)
            JOIN propertylandusetype as proptype USING(propertylandusetypeid)
            WHERE (pred.transactiondate between '2017-05-01'
							AND '2017-08-31')
					AND (prop.propertylandusetypeid IN (260, 261, 262, 263, 264,
                                                         268, 269, 273, 274, 275, 276, 279));
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