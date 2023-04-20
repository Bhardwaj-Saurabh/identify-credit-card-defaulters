
from src.entity.config_entity import TrainingPipelineConfig,DataIngestionConfig,DataValidationConfig
from src.entity.config_entity import DataPreparationConfig, DataTransformationConfig
from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact, DataPreprationArtifact
from src.entity.artifact_entity import DataTransformationArtifact
from src.exception import CustomException
import sys
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_prepration import DataPrepration
from src.components.data_transformation import DataTransformation

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
        
    def start_data_preprationtion(self, data_validation_artifact:DataValidationArtifact) -> DataPreprationArtifact:
        """
        Starts the data preparation process for the training pipeline.

        Args:
            data_validation_artifact (DataValidationArtifact): Artifact of the data validation process.

        Returns:
            data_preparation_artifact (DataPreparationArtifact): Artifact of the data preparation process.
        """
        try:
            data_prepration_config = DataPreparationConfig(training_pipeline_config=self.training_pipeline_config)
            data_prepration = DataPrepration(data_validation_artifact=data_validation_artifact,
                                             data_prepration_config=data_prepration_config)
            data_prepration_artifact = data_prepration.initiate_data_prepration()
            return data_prepration_artifact
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_data_transformation(self, data_prepration_artifact:DataPreprationArtifact) -> DataTransformationArtifact:
        """
        Starts the data transformation process for the training pipeline.

        Args:
            data_preparation_artifact (DataPreparationArtifact): Artifact of the data preparation process.

        Returns:
            data_transformation_artifact (DataTransformationArtifact): Artifact of the data transformation process.
        """
        try:
            data_transformation_config = DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            data_transformation = DataTransformation(data_prepration_artifact=data_prepration_artifact,
                                                    data_transformation_config=data_transformation_config)
            data_transformation_artifact =  data_transformation.initiate_data_transformation()
            return data_transformation_artifact
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
            data_preparation_artifact = self.start_data_preprationtion(data_validation_artifact=data_validation_artifact)     
            data_transformation_artifact = self.start_data_transformation(data_prepration_artifact=data_preparation_artifact)  
        except  Exception as e:
            raise  CustomException(e,sys)