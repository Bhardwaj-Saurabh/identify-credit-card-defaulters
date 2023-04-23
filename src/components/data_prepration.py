
from src.entity.artifact_entity import DataValidationArtifact, DataPreparationArtifact
from src.entity.config_entity import DataPreparationConfig
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from pandas import DataFrame
import numpy as np
import os,sys
from typing import List
from src.utils.main_utils import write_yaml_file, reduce_mem_usage
from scipy.stats import ks_2samp

class DataPreparation:
    def __init__(self, data_validation_artifact: DataValidationArtifact, data_preparation_config: DataPreparationConfig):
        """
        Initialize the DataPreparation class.

        Args:
            data_validation_artifact (DataValidationArtifact): Output reference of data validation artifact stage
            data_preparation_config (DataPreparationConfig): Configuration for data preparation
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_preparation_config = data_preparation_config
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Read data from file path and return a pandas DataFrame.

        Args:
            file_path (str): File path to read data from.

        Returns:
            pd.DataFrame: Pandas DataFrame containing the read data.
        """
        try:
            return pd.read_csv(file_path, nrows=25000)
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform data preprocessing on a given pandas DataFrame.

        Args:
            df (pd.DataFrame): Pandas DataFrame to be preprocessed.

        Returns:
            pd.DataFrame: Preprocessed pandas DataFrame.
        """
        try:
            logging.info("Adding missing values columns with missing flag")
            # Add flag column for missing values
            for col in df.columns:
                df[col+"_missing_flag"] = df[col].isnull()
            logging.info("Missing values columns added")
            return df
        except Exception as e:
            raise CustomException(e, sys)

        
    @staticmethod
    def get_list_of_columns_to_drop(df: pd.DataFrame) -> List[str]:
        """
        Get the list of column names to drop from a given pandas DataFrame based on criteria such as missing values and
        standard deviation.

        Args:
            df (pd.DataFrame): Pandas DataFrame to analyze.

        Returns:
            List[str]: List of column names to keep in the DataFrame.
        """
        try:
            logging.info("Getting the column names with more than 90% missing values")
            # Drop the columns where one category contains more than 90% values
            drop_cols = []
            for col in df.columns:
                missing_share = df[col].isnull().sum() / df.shape[0]
                if missing_share > 0.9:
                    drop_cols.append(col)
            column_to_keep = [col for col in df.columns if col not in drop_cols]

            logging.info("Column names with more than 90% missing values collected")

            logging.info("Getting the column names with zero standard deviation")
            # Drop the columns which have only one unique value
            drop_cols = []
            for col in column_to_keep:
                unique_value = df[col].nunique()
                if unique_value == 1:
                    drop_cols.append(col)

            column_to_keep = [col for col in column_to_keep if col not in drop_cols]
            logging.info("Column names with zero standard deviation collected")

            logging.info(f"Columns to keep in the dataset are: {column_to_keep}")
            return column_to_keep
        except Exception as e:
            raise CustomException(e, sys)

    
    def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
        try:
            status=True
            report ={}
            for column in base_df.columns:
                d1 = base_df[column]
                d2  = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold<=is_same_dist.pvalue:
                    is_found=False
                else:
                    is_found = True 
                    status=False
                report.update({column:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status":is_found
                    
                    }})
            
            drift_report_file_path = self.data_prepration_config.drift_report_file_path
            
            #Create directory
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report,)
            return status
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def create_domain_specific_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific features based on transaction amount and card information in a given pandas DataFrame.

        Args:
            df (pd.DataFrame): Pandas DataFrame to create features on.

        Returns:
            pd.DataFrame: DataFrame with domain-specific features added.
        """
        try:
            logging.info("Creating features based on transaction amount")
            # Transaction amount minus mean of transaction
            df['TransactionAmt_minus_mean'] = df['TransactionAmt'] - np.nanmean(df['TransactionAmt'], dtype="float64")
            df['TransactionAmt_minus_std'] = df['TransactionAmt_minus_mean'] / np.nanstd(df['TransactionAmt'].astype("float64"), dtype="float64")
            df['TransactionAmt'] = np.log(df['TransactionAmt'])

            logging.info("Creating features for transaction amount and card")
            # Features for transaction amount and card
            df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')

            df['TransactionAmt_to_std_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
            df['TransactionAmt_to_std_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')

            return df
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_preparation(self) -> DataPreparationArtifact:
        """
        Initiates the data preparation process by reading, preprocessing, creating domain-specific features,
        reducing memory usage, and saving the prepared data to a CSV file.

        Returns:
            DataPreparationArtifact: Data preparation artifact containing file paths for prepared data and drift report.
        """
        try:
            file_path = self.data_validation_artifact.valid_file_path
            dataframe = DataPreparation.read_data(file_path)
            dataframe = DataPreparation.preprocess_data(dataframe)
            columns_to_keep = DataPreparation.get_list_of_columns_to_drop(dataframe)
            dataframe = dataframe[columns_to_keep]
            print(dataframe.shape)
            dataframe = DataPreparation.create_domain_specific_features(dataframe)
            dataframe = reduce_mem_usage(dataframe)
            logging.info(f"Final shape of the data is {dataframe.shape}")

            logging.info("Creating prepared dataset directory to store prepared data")
            dir_path = os.path.dirname(self.data_preparation_config.prepared_data_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Saving prepared dataset")
            dataframe.to_csv(
                self.data_preparation_config.prepared_data_file_path, index=False, header=True
            ) # Save prepared data to CSV file

            data_preparation_artifact = DataPreparationArtifact(
                prepared_data_file_path=self.data_preparation_config.prepared_data_file_path,
                drift_report_file_path=self.data_preparation_config.drift_report_file_path,
            ) # Create data preparation artifact

            logging.info(f"Data preparation artifact: {data_preparation_artifact}")
            return data_preparation_artifact
        except Exception as e:
            raise CustomException(e, sys)