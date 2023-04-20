
from src.entity.artifact_entity import DataValidationArtifact, DataPreprationArtifact
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

class DataPrepration:

    def __init__(self,data_validation_artifact:DataValidationArtifact,
                        data_prepration_config:DataPreparationConfig):
        try:
            self.data_validation_artifact=data_validation_artifact
            self.data_prepration_config=data_prepration_config
        except Exception as e:
            raise  CustomException(e,sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def preprocess_data(df: pd.DataFrame)->pd.DataFrame:
        """
        Perform data preprocessing
        """
        try:
            logging.info("Adding missing values columns with missing flag")
            # Add flag column for missing values
            for col in df.columns:
                df[col+"_missing_flag"] = df[col].isnull()
            logging.info("Missing values columns added")
            return df
        except  Exception as e:
            raise  CustomException(e,sys)
        
    @staticmethod
    def get_list_of_columns_to_drop(df:pd.DataFrame)->List[str]:
        try:
            logging.info("Getting the coloumn names with more than 90% missing values")
            # Drop the columns where one category contains more than 90% values
            drop_cols = []
            for col in df.columns:
                missing_share = df[col].isnull().sum()/df.shape[0]
                if missing_share > 0.9:
                    drop_cols.append(col)        
            column_to_keep = [col for col in df.columns if col not in drop_cols]    

            logging.info("coloumn names with more than 90% missing values collected")

            logging.info("Getting the coloumn names with zero standard deviation")
            # Drop the columns which have only one unique value
            drop_cols = []
            for col in column_to_keep:
                unique_value = df[col].nunique()
                if unique_value == 1:
                    drop_cols.append(col)

            column_to_keep = [col for col in column_to_keep if col not in drop_cols]
            logging.info("coloumn names with Zero standard deviation collected")

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
    def create_domain_specific_features(df: DataFrame)-> DataFrame:
        try:
            logging.info(" Create feature based on Transaction amount based feature")
            # Transaction amount minus mean of transaction 
            df['Trans_min_mean'] = df['TransactionAmt'] - np.nanmean(df['TransactionAmt'],dtype="float64")
            df['Trans_min_std']  = df['Trans_min_mean'] / np.nanstd(df['TransactionAmt'].astype("float64"),dtype="float64")
            df['TransactionAmt'] = np.log(df['TransactionAmt'])

            logging.info(" Creating Features for transaction amount and card ")
           # Features for transaction amount and card 
            df['TransactionAmt_to_mean_card1'] = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('mean')
            df['TransactionAmt_to_mean_card4'] = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('mean')
            
            df['TransactionAmt_to_std_card1']  = df['TransactionAmt'] / df.groupby(['card1'])['TransactionAmt'].transform('std')
            df['TransactionAmt_to_std_card4']  = df['TransactionAmt'] / df.groupby(['card4'])['TransactionAmt'].transform('std')
            return df
        except Exception as e:
            raise CustomException(e, sys)


    def initiate_data_prepration(self)->DataPreprationArtifact:
        try:
            file_path = self.data_validation_artifact.valid_file_path
            dataframe = DataPrepration.read_data(file_path)
            dataframe = DataPrepration.preprocess_data(dataframe)

            column_to_keep = DataPrepration.get_list_of_columns_to_drop(dataframe)
            dataframe = dataframe[column_to_keep]
            dataframe = DataPrepration.create_domain_specific_features(dataframe)
            dataframe = reduce_mem_usage(dataframe)
            logging.info(f"Final shape of the data is {dataframe.shape}")

            logging.info("Creating prepared dataset directory to store prepared data")
            dir_path = os.path.dirname(self.data_prepration_config.prepared_data_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Prepared dataset saved successfully")
            dataframe.to_csv(
                self.data_prepration_config.prepared_data_file_path, index=False, header=True
            )

            data_prepration_artifact = DataPreprationArtifact(
                prepared_data_file_path=self.data_prepration_config.prepared_data_file_path,
                drift_report_file_path=self.data_prepration_config.drift_report_file_path,
            )

            logging.info(f"Data prepration artifact: {data_prepration_artifact}")
            return data_prepration_artifact
        except Exception as e:
            raise CustomException(e,sys)
