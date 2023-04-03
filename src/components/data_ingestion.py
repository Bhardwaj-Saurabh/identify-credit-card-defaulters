from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

import os,sys
from pandas import DataFrame
from src.data_access.credit_card_data import CreditCardData

class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config
        except Exception as e:
            raise CustomException(e,sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export mongo db collection record as data frame into feature
        """
        try:
            logging.info("Exporting data from mongodb to feature store")
            credit_card_data = CreditCardData()
            dataframe = credit_card_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path            

            #creating folder
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
        except  Exception as e:
            raise  CustomException(e,sys)
    
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.export_data_into_feature_store()
            data_ingestion_artifact = DataIngestionArtifact(
                ingested_file_path=self.data_ingestion_config.feature_store_file_path
                )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e,sys)