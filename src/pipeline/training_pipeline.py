
from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig
from src.entity.config_entity import DataValidationConfig, DataPreprationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.exception import CustomException
import sys,os
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_prepration import DataPrepration

class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self)-> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_data_validaton(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  CustomException(e,sys)
        
    def start_data_preprationtion(self, data_validation_artifact:DataValidationArtifact):
        try:
            data_prepration_config = DataPreprationConfig(training_pipeline_config=self.training_pipeline_config)
            data_prepration = DataPrepration(data_validation_artifact=data_validation_artifact,
                                             data_prepration_config=data_prepration_config)
            data_prepration_artifact = data_prepration.initiate_data_prepration()
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_data_transformation(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)
        
    
    def start_model_trainer(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_model_evaluation(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_model_pusher(self):
        try:
           pass
        except  Exception as e:
            raise  CustomException(e,sys)


    def run_pipeline(self):
        try:
            data_ingestion_artifact:DataIngestionArtifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validaton(data_ingestion_artifact=data_ingestion_artifact)
            data_prepration_artifact = self.start_data_preprationtion(data_validation_artifact=data_validation_artifact)       
        except  Exception as e:
            raise  CustomException(e,sys)