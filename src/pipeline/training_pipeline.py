
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_prepration import DataPreparation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig
from src.entity.config_entity import DataPreparationConfig, DataTransformationConfig, ModelTrainerConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataPreparationArtifact
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact

class TrainPipeline:
    """
    Class to represent a training pipeline for a machine learning model.
    """

    # Class-level attribute to track whether the pipeline is running or not
    is_pipeline_running = False

    def __init__(self):
        """
        Initializes the training pipeline.

        Attributes:
            training_pipeline_config (TrainingPipelineConfig): Configuration for the training pipeline.
        """
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        """
        Starts the data ingestion process for the training pipeline.

        Returns:
            data_ingestion_artifact (DataIngestionArtifact): Artifact of the data ingestion process.
        """
        try:
            self.data_ingestion_config = DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)


    def start_data_validaton(self, data_ingestion_artifact:DataIngestionArtifact)->DataValidationArtifact:
        """
        Starts the data validation process for the training pipeline.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): Artifact of the data ingestion process.

        Returns:
            data_validation_artifact (DataValidationArtifact): Artifact of the data validation process.
        """
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,
            data_validation_config = data_validation_config
            )
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except  Exception as e:
            raise  CustomException(e,sys)
        
    def start_data_preparationtion(self, data_validation_artifact:DataValidationArtifact) -> DataPreparationArtifact:
        """
        Starts the data preparation process for the training pipeline.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact of the data validation process.

        Returns:
            data_preparation_artifact (DataPreparationArtifact): Artifact of the data preparation process.
        """
        try:
            data_preparation_config = DataPreparationConfig(training_pipeline_config=self.training_pipeline_config)
            data_preparation = DataPreparation(data_validation_artifact=data_validation_artifact,
                                             data_preparation_config=data_preparation_config)
            data_preparation_artifact = data_preparation.initiate_data_preparation()
            return data_preparation_artifact
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_data_transformation(self, data_preparation_artifact:DataPreparationArtifact) -> DataTransformationArtifact:
        """
        Starts the data transformation process for the training pipeline.

        Args:
            data_preparation_artifact (DataPreparationArtifact): Artifact of the data preparation process.

        Returns:
            data_transformation_artifact (DataTransformationArtifact): Artifact of the data transformation process.
        """
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_preparation_artifact=data_preparation_artifact,
                                                    data_transformation_config=data_transformation_config)
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except  Exception as e:
            raise  CustomException(e,sys)
        
    def start_model_trainer(self, data_transformation_artifact:DataTransformationArtifact) -> ModelTrainerArtifact:
        """
        Starts the model trainer process for the training pipeline.

        Args:
            data_transformation_artifact (DataTransformationArtifact): Artifact of the data transformation process.

        Returns:
            model_trainer_artifact (ModelTrainerArtifact): Artifact of the model trainer process.
        """
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config=model_trainer_config,
                                         data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            return model_trainer_artifact
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
            data_preparation_artifact = self.start_data_preparationtion(data_validation_artifact=data_validation_artifact)     
            data_transformation_artifact = self.start_data_transformation(data_preparation_artifact=data_preparation_artifact)  
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
        except  Exception as e:
            raise  CustomException(e,sys)