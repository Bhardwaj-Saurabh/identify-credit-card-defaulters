
from src.entity.artifact_entity import DataValidationArtifact, DataPreprationArtifact
from src.entity.config_entity import DataPreprationConfig
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import os,sys
from typing import List
from sklearn.model_selection import train_test_split
from src.utils.main_utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp

class DataPrepration:

    def __init__(self,data_validation_artifact:DataValidationArtifact,
                        data_prepration_config:DataPreprationConfig):
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
   
    def preprocess_data(self, dataframe: pd.DataFrame)->pd.DataFrame:
        """
        Perform data preprocessing
        """
        try:
            logging.info("Adding missing values columns with missing flag")
            # Add flag column for missing values
            for col in dataframe.columns:
                dataframe[col+"_missing_flag"] = dataframe[col].isnull()
            logging.info("Missing values columns added")
            return dataframe
        except  Exception as e:
            raise  CustomException(e,sys)
        
    @staticmethod
    def get_list_of_columns_to_drop(dataframe:pd.DataFrame)->List['str']:
        try:
            logging.info("Getting the coloumn names with more than 90% missing values")
            # Drop the columns where one category contains more than 90% values
            drop_cols = []
            for col in dataframe.columns:
                missing_share = dataframe[col].isnull().sum()/dataframe.shape[0]
                if missing_share > 0.9:
                    drop_cols.append(col)        
            column_to_keep = [col for col in dataframe.columns if col not in drop_cols]    

            logging.info("coloumn names with more than 90% missing values collected")

            logging.info("Getting the coloumn names with zero standard deviation")

            # Drop the columns which have only one unique value
            drop_cols = []
            for col in column_to_keep:
                unique_value = dataframe[col].nunique()
                if unique_value == 1:
                    drop_cols.append(col)
            column_to_keep = [col for col in column_to_keep if col not in drop_cols]

            logging.info("coloumn names with Zero standard deviation collected")

            logging.info(f"Columns to keep in the dataset are: {column_to_keep}")
            return column_to_keep
        except Exception as e:
            raise CustomException(e, sys)
        
    def split_data_as_train_test(self, dataframe: pd.DataFrame) -> None:
        """
        Feature store dataset will be split into train and test file
        """

        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_prepration_config.prepared_train_test_split_ratio
            )
            logging.info("Performed train test split on the dataframe")
            logging.info(
                "Exited split_data_as_train_test method of Data_Ingestion class"
            )

            dir_path = os.path.dirname(self.data_prepration_config.prepared_train_data_file_path)
            print(dir_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info(f"Exporting train and test file path.")
            train_set.to_csv(
                self.data_prepration_config.prepared_train_data_file_path, index=False, header=True
            )

            test_set.to_csv(
                self.data_prepration_config.prepared_test_data_file_path, index=False, header=True
            )

            logging.info(f"Exported train and test file path.")
        except Exception as e:
            raise CustomException(e,sys)
        

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

    def initiate_data_prepration(self)->DataPreprationArtifact:
        try:
            file_path = self.data_validation_artifact.valid_file_path
            dataframe = DataPrepration.read_data(file_path)
            dataframe = self.preprocess_data(dataframe)

            column_to_keep = DataPrepration.get_list_of_columns_to_drop(dataframe)
            dataframe = dataframe[column_to_keep]

            self.split_data_as_train_test(dataframe)

            # #Let check data drift
            # prepared_train_file_path=self.data_prepration_config.prepared_train_data_file_path
            # prepared_test_file_path=self.data_prepration_config.prepared_test_data_file_path

            # train_dataframe = DataPrepration.read_data(prepared_train_file_path)
            # test_dataframe = DataPrepration.read_data(prepared_test_file_path)

            #status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            data_prepration_artifact = DataPreprationArtifact(
                validation_status=None,
                prepared_train_file_path=self.data_prepration_config.prepared_train_data_file_path,
                prepared_test_file_path=self.data_prepration_config.prepared_test_data_file_path,
                drift_report_file_path=self.data_prepration_config.drift_report_file_path,
            )

            logging.info(f"Data prepration artifact: {data_prepration_artifact}")

            return data_prepration_artifact
        except Exception as e:
            raise CustomException(e,sys)
