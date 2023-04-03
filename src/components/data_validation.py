from distutils import dir_util
from src.constant.training_pipeline import SCHEMA_FILE_PATH
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import read_yaml_file,write_yaml_file
from scipy.stats import ks_2samp
import pandas as pd
import os,sys
class DataValidation:

    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                        data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact=data_ingestion_artifact
            self.data_validation_config=data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise  CustomException(e,sys)


    def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Data frame has columns: {len(dataframe.columns)}")
            if len(dataframe.columns)==number_of_columns:
                return True
            return False
        except Exception as e:
            raise CustomException(e,sys)

    def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
        try:
            numerical_columns = self._schema_config["numerical_columns"]
            dataframe_columns = dataframe.columns

            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:
                    numerical_column_present=False
                    missing_numerical_columns.append(num_column)
            
            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
            return numerical_column_present
        except Exception as e:
            raise CustomException(e,sys)

    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
   

    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            error_message = ""
            ingested_file_path = self.data_ingestion_artifact.ingested_file_path

            #Reading data from train and test file location
            dataframe = DataValidation.read_data(ingested_file_path)

            #Validate number of columns
            status = self.validate_number_of_columns(dataframe=dataframe)
            if not status:
                error_message=f"{error_message}Dataframe does not contain all columns.\n"

            #Validate numerical columns

            status = self.is_numerical_column_exist(dataframe=dataframe)
            if not status:
                error_message=f"{error_message}Train dataframe does not contain all numerical columns.\n"
            
            
            if len(error_message)>0:
                raise Exception(error_message)

            # #Let check data drift
            # status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

            data_validation_artifact = DataValidationArtifact(
                valid_file_path=self.data_ingestion_artifact.ingested_file_path,
                invalid_file_path=None
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)
