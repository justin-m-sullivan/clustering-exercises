import pandas as pd
import numpy as np
import os
from env import host, user, password
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Statistical Tests
import scipy.stats as stats

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

def get_latitude(df):
    '''
    This function takes in a datafame with latitude formatted as a float,
    converts it to a int and utilizes lambda to return the latitude values
    in a correct format.
    '''
    df.latitude = df.latitude.astype(int)
    df['latitude'] = df['latitude'].apply(lambda x: x / 10 ** (len((str(x))) - 2))
    return df

def get_longitude(df):
    '''This function takes in a datafame with longitude formatted as a float,
    converts it to a int and utilizes lambda to return the longitude values
    in the correct format.
    '''
    df.longitude = df.longitude.astype(int)
    df['longitude'] = df['longitude'].apply(lambda x: x / 10 ** (len((str(x))) - 4))
    return df

def clean_zillow(df):
    df = get_single_use_prop(df)

    df = handle_missing_values(df, prop_required_row = 0.5, prop_required_col = 0.5)

    df.set_index('parcelid', inplace=True)

    cols_to_drop = ['fullbathcnt','heatingorsystemtypeid','finishedsquarefeet12', 
                'propertycountylandusecode', 'propertylandusetypeid','propertyzoningdesc', 'censustractandblock',
                'propertylandusedesc', 'buildingqualitytypeid' , 'unitcnt', 'heatingorsystemdesc', 
                'lotsizesquarefeet','regionidcity', 'calculatedbathnbr', 'transactiondate', 'roomcnt', 'id', 'regionidcounty',
                'regionidzip', 'assessmentyear']

    df.drop(columns=cols_to_drop, inplace = True)

    df.dropna(inplace = True)

    get_latitude(df)

    get_longitude(df)

    return df

def get_county(df):
    #Convert fips to int
    df.fips = df.fips.astype('int64')

    county = []

    for row in df['fips']:
        if row == 6037:
            county.append('Los Angeles')
        elif row == 6059:
            county.append('Orange')
        elif row == 6111:
            county.append('Ventura')
        
    df['county'] = county

    df.drop(columns={'fips'}, inplace=True)
    return df

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

def train_validate_test_split(df, target, seed=123):
    '''
    This function takes in a dataframe, the name of the target variable
    (for stratification purposes), and an integer for a setting a seed
    and splits the data into train, validate and test. 
    Test is 20% of the original dataset, validate is .30*.80= 24% of the 
    original dataset, and train is .70*.80= 56% of the original dataset. 
    The function returns, in this order, train, validate and test dataframes. 
    '''
    train_validate, test = train_test_split(df, test_size=0.2, 
                                            random_state=seed, 
                                            stratify=df[target])
    train, validate = train_test_split(train_validate, test_size=0.3, 
                                       random_state=seed,
                                       stratify=train_validate[target])
    return train, validate, test

def scale_my_data(train, validate, test):
    #call numeric cols
    numeric_cols = get_numeric_cols(df)
    scaler = StandardScaler()
    scaler.fit(train[[numeric_cols]])

    X_train_scaled = scaler.transform(train[[numeric_cols]])
    X_validate_scaled = scaler.transform(validate[[numeric_cols]])
    X_test_scaled = scaler.transform(test[[numeric_cols]])

    train[[numeric_cols]] = X_train_scaled
    validate[[numeric_cols]] = X_validate_scaled
    test[[numeric_cols]] = X_test_scaled
    return train, validate, test

def prepare_zillow(df):
    #Separate logerror into quantiles
    df['logerror_class'] = pd.qcut(df.logerror, q=4, labels=['q1', 'q2', 'q3', 'q4'])
    get_county(df)
    return df

    #Split data into Train, Validate, and Test
    #train, validate, test = train_validate_test_split(df, target='logerror_class', seed=123)
    #train, validate, test = scale_my_data(train, validate, test)

    #return train, validate, test