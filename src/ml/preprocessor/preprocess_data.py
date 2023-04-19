from src.exception import CustomException
from src.logger import logging
import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.decomposition import PCA


def perform_PCA(df, cols, n_components, prefix='PCA_', rand_seed=4):
    try:
        pca = PCA(n_components=n_components, random_state=rand_seed)
        principalComponents = pca.fit_transform(df[cols])
        principalDf = pd.DataFrame(principalComponents)
        
        df.drop(cols, axis=1, inplace=True)

        principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)
        df = pd.concat([df, principalDf], axis=1)
        return df
    except Exception as e:
        raise CustomException(e, sys)
        

def missing_values_and_scalling_encoder(df, filter_col):
    try:
        # Fill na values and scale V columns
        for col in filter_col:
            df[col] = df[col].fillna((df[col].min() - 2))
            df[col] = (minmax_scale(df[col], feature_range=(0,1)))
        return df
    except Exception as e:
        raise CustomException(e, sys)


def frequency_encoder(df):
    try:
        cat_columns = df.select_dtypes(include=['object']).columns
        binary_columns = [col for col in df.columns if df[col].nunique() == 2]        

        cat_columns = cat_columns.to_list() + binary_columns
        # Frequecny encoding variables
        frequency_encoded_variables = []
        for col in cat_columns:
            if df[col].nunique() > 30:
                print(col, df[col].nunique())
                frequency_encoded_variables.append(col)

        # Frequecny enocde the variables
        for variable in frequency_encoded_variables:
            # group by frequency 
            fq = df.groupby(variable).size()/len(df)    
            # mapping values to dataframe 
            df.loc[:, "{}".format(variable)] = df[variable].map(fq)   
            cat_columns.remove(variable)

        # Label encode the variables
        for col in cat_columns:
            lbl = LabelEncoder()
            lbl.fit(list(df[col].values))
            df[col] = lbl.transform(list(df[col].values))
        return df
    except Exception as e:
        raise CustomException(e, sys)
