from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.data_access.credit_card_data import CreditCardData
from src.utils.main_utils import reduce_mem_usage

import os
import sys
from pandas import DataFrame


class DataIngestion:
    """
    Class to represent data ingestion for the training pipeline.
    """

    def __init__(self, data_ingestion_config: DataIngestionConfig):
        """
        Initializes the data ingestion process.

        Args:
            data_ingestion_config (DataIngestionConfig): Configuration for the data ingestion process.
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise CustomException(e, sys)

    def export_data_into_feature_store(self) -> DataFrame:
        """
        Export MongoDB collection records as a DataFrame into the feature store.
        """
        try:
            logging.info("Exporting data from MongoDB to feature store")
            credit_card_data = CreditCardData()
            dataframe = credit_card_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            dataframe = reduce_mem_usage(dataframe)
            print(dataframe.shape)

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path

            # Creating folder if not exists
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False, header=True)
            logging.info("Data export completed")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates the data ingestion process and returns the data ingestion artifact.

        Returns:
            data_ingestion_artifact (DataIngestionArtifact): Artifact of the data ingestion process.
        """
        try:
            self.export_data_into_feature_store()
            data_ingestion_artifact = DataIngestionArtifact(
                ingested_file_path=self.data_ingestion_config.feature_store_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)
