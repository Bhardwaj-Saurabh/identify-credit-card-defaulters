from src.exception import CustomException
from src.logger import logging
from src.constant.training_pipeline import RANDOM_SEED
import pandas as pd
import sys

from sklearn.preprocessing import LabelEncoder, minmax_scale
from sklearn.decomposition import PCA

# Function to Perform Principal Component Analysis
def perform_PCA(dataframe, columns, n_components, prefix='PCA_', rand_seed=RANDOM_SEED):
    """
    Perform Principal Component Analysis (PCA) on the given dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        columns (list): List of column names to perform PCA on.
        n_components (int): Number of principal components to keep.
        prefix (str, optional): Prefix for column names of principal components. Defaults to 'PCA_'.
        rand_seed (int, optional): Random seed for reproducibility. Defaults to 4.

    Returns:
        pd.DataFrame: Dataframe with PCA components appended.

    Raises:
        CustomException: If any error occurs during PCA.
    """
    try:
        logging.info("Performing PCA...") # Log message to indicate start of PCA
        pca = PCA(n_components=n_components, random_state=rand_seed)
        pca_components = pca.fit_transform(dataframe[columns])
        principalDf = pd.DataFrame(pca_components)
        
        dataframe.drop(columns, axis=1, inplace=True)

        principalDf.rename(columns=lambda x: str(prefix)+str(x), inplace=True)
        dataframe = pd.concat([dataframe, principalDf], axis=1)

        logging.info("PCA completed successfully.") # Log message to indicate successful completion of PCA
        return dataframe
    except Exception as e:
        raise CustomException(e, sys)

# Function to fill missing values and scale columns.
def missing_values_and_scaling_encoder(dataframe, columns):
    """
    Fill missing values and scale columns in the given dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.
        columns (list): List of column names to fill missing values and scale.

    Returns:
        pd.DataFrame: Dataframe with missing values filled and columns scaled.

    Raises:
        CustomException: If any error occurs during missing value filling or scaling.
    """
    try:
        logging.info("Filling missing values and scaling columns...") # Log message to indicate start of operation
        for col in columns:
            # Fill missing values with the minimum value minus 2
            dataframe[col] = dataframe[col].fillna((dataframe[col].min() - 2))
            # Scale the column using min-max scaling to range from 0 to 1
            dataframe[col] = minmax_scale(dataframe[col], feature_range=(0, 1))

        logging.info("Missing value filling and scaling completed successfully.") # Log message to indicate successful completion
        return dataframe
    except Exception as e:
        raise CustomException(e, sys)

# Function to perform frequency encoding and label encoding 
def frequency_encoder(dataframe):
    """
    Perform frequency encoding and label encoding on categorical columns in the given dataframe.

    Args:
        dataframe (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with frequency encoded and label encoded columns.

    Raises:
        CustomException: If any error occurs during frequency encoding or label encoding.
    """
    try:
        logging.info("Performing frequency encoding and label encoding on categorical columns...") # Log message to indicate start of operation
        cat_columns = dataframe.select_dtypes(include=['object']).columns
        binary_columns = [col for col in dataframe.columns if dataframe[col].nunique() == 2]
        cat_columns = cat_columns.to_list() + binary_columns

        frequency_encoded_variables = []
        for col in cat_columns:
            if dataframe[col].nunique() > 30:
                print(col, dataframe[col].nunique())
                frequency_encoded_variables.append(col)

        for variable in frequency_encoded_variables:
            fq = dataframe.groupby(variable).size() / len(dataframe)
            dataframe.loc[:, "{}".format(variable)] = dataframe[variable].map(fq)
            cat_columns.remove(variable)

        for col in cat_columns:
            lbl = LabelEncoder()
            lbl.fit(list(dataframe[col].values))
            dataframe[col] = lbl.transform(list(dataframe[col].values))

        logging.info("Frequency encoding and label encoding completed successfully.") # Log message to indicate successful completion
        return dataframe
    except Exception as e:
        raise CustomException(e, sys)

