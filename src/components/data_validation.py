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
    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Constructor for DataValidation class.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Data ingestion artifact containing file path of the
                                                            ingested data.
            data_validation_config (DataValidationConfig): Data validation configuration containing settings for
                                                          data validation.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact  # Store data ingestion artifact
            self.data_validation_config = data_validation_config  # Store data validation configuration
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)  # Read schema configuration from YAML file
        except Exception as e:
            raise CustomException(e, sys)


    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate the number of columns in the given dataframe against the schema configuration.

        Args:
            dataframe (pd.DataFrame): Dataframe to be validated.

        Returns:
            bool: True if the number of columns in the dataframe matches the required number of columns from the
                  schema configuration, False otherwise.
        """
        try:
            number_of_columns = len(self._schema_config["columns"])  # Get the required number of columns from schema config
            logging.info(f"Required number of columns: {number_of_columns}")
            logging.info(f"Dataframe has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == number_of_columns:  # Compare number of columns in dataframe with required number of columns
                return True
            return False
        except Exception as e:
            raise CustomException(e, sys)


    def is_numerical_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Check if all the required numerical columns are present in the given dataframe based on the schema configuration.

        Args:
            dataframe (pd.DataFrame): Dataframe to be checked.

        Returns:
            bool: True if all the required numerical columns are present in the dataframe, False otherwise.
        """
        try:
            numerical_columns = self._schema_config["numerical_columns"]  # Get the list of required numerical columns from schema config
            dataframe_columns = dataframe.columns  # Get the columns in the dataframe

            numerical_column_present = True
            missing_numerical_columns = []
            for num_column in numerical_columns:
                if num_column not in dataframe_columns:  # Check if required numerical column is missing in the dataframe
                    numerical_column_present = False
                    missing_numerical_columns.append(num_column)  # Append the missing numerical column to the list

            logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
            return numerical_column_present
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read data from a CSV file and return as a DataFrame.

        Args:
            file_path (str): File path of the CSV file.

        Returns:
            pd.DataFrame: DataFrame containing the data from the CSV file.
        """
        try:
            return pd.read_csv(file_path)  # Read data from the CSV file using pandas read_csv method
        except Exception as e:
            raise CustomException(e, sys)

   
    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Initiate data validation process.

        Returns:
            DataValidationArtifact: DataValidationArtifact object containing the valid and invalid file paths.
        """
        try:
            error_message = ""  # Initialize error message

            ingested_file_path = self.data_ingestion_artifact.ingested_file_path  # Get ingested file path from data ingestion artifact

            # Reading data from ingested file path
            dataframe = DataValidation.read_data(ingested_file_path)

            # Validate number of columns
            status = self.validate_number_of_columns(dataframe=dataframe)
            if not status:
                error_message = f"{error_message}Dataframe does not contain all columns.\n"

            # Validate numerical columns
            status = self.is_numerical_column_exist(dataframe=dataframe)
            if not status:
                error_message = f"{error_message}Train dataframe does not contain all numerical columns.\n"

            if len(error_message) > 0:
                raise Exception(error_message)

            # Create data validation artifact
            data_validation_artifact = DataValidationArtifact(
                valid_file_path=self.data_ingestion_artifact.ingested_file_path,
                invalid_file_path=None
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
