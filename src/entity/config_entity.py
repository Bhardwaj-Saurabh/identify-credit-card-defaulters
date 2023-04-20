
from datetime import datetime
import os
from src.constant  import training_pipeline

class TrainingPipelineConfig:
    def __init__(self, timestamp: datetime = datetime.now()):
        """
        Initialize the TrainingPipelineConfig class.

        Args:
            timestamp (datetime, optional): The timestamp for the training pipeline, defaults to the current datetime.
        """
        timestamp_str = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp_str)
        self.timestamp: str = timestamp_str

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize the DataIngestionConfig class.

        Args:
            training_pipeline_config (TrainingPipelineConfig): The TrainingPipelineConfig object used to generate data ingestion configuration.
        """
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize the DataValidationConfig class.

        Args:
            training_pipeline_config (TrainingPipelineConfig): The TrainingPipelineConfig object used to generate data validation configuration.
        """
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.FILE_NAME)
        self.invalid_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.FILE_NAME)
        
class DataPreparationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize the DataPreparationConfig class.

        Args:
            training_pipeline_config (TrainingPipelineConfig): The TrainingPipelineConfig object used to generate data preparation configuration.
        """
        self.data_preparation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_PREPARATION_DIR_NAME
        )
        self.prepared_data_file_path: str = os.path.join(self.data_preparation_dir, training_pipeline.FILE_NAME)
        self.drift_report_file_path: str = os.path.join(
            self.data_preparation_dir,
            training_pipeline.DATA_PREPARATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_PREPARATION_DRIFT_REPORT_FILE_NAME,
        )
class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize the DataTransformationConfig class.

        Args:
            training_pipeline_config (TrainingPipelineConfig): The TrainingPipelineConfig object used to generate data transformation configuration.
        """
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        self.transformed_data_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
            training_pipeline.FILE_NAME.replace("csv", "npy"),
        )