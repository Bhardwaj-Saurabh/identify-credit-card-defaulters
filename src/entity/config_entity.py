
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
        # Directory path for the ingested data
        self.data_ingestion_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
        )
        # File path for the feature store
        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
        )
        # Collection name config
        self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize the DataValidationConfig class.

        Args:
            training_pipeline_config (TrainingPipelineConfig): The TrainingPipelineConfig object used to generate data validation configuration.
        """
        # Directory path for the valid data
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        # File path for the valida data file path
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
        # Directory path for the data preparation
        self.data_preparation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_PREPARATION_DIR_NAME
        )
        # File path for the prepared file path
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
        # Directory path for the data transformation
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME
        )
        # File path for the transformed file path
        self.transformed_data_file_path: str = os.path.join(
            self.data_transformation_dir,
            training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
        )
        # File path for the transformed train data file path
        self.transformed_train_data_file_path: str = os.path.join(
            self.transformed_data_file_path,
            training_pipeline.TRAIN_FILE_NAME.replace("csv", "npy"),
        )
        # File path for the transformed test data file path
        self.transformed_test_data_file_path: str = os.path.join(
            self.transformed_data_file_path,
            training_pipeline.TEST_FILE_NAME.replace("csv", "npy"),
        )

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Class for model trainer configuration.

        Args:
            training_pipeline_config (TrainingPipelineConfig): Configuration object for training pipeline.
        """
        # Directory path for the model trainer
        self.model_trainer_dir: str = os.path.join(training_pipeline_config.artifact_dir,
                                                  training_pipeline.MODEL_TRAINER_DIR_NAME)
        # File path for the trained model
        self.trained_model_file_path: str = os.path.join(self.model_trainer_dir,
                                                         training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR,
                                                         training_pipeline.MODEL_FILE_NAME)
        # Expected accuracy for the trained model
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINER_EXPECTED_SCORE
        # train test split ratio
        self.train_test_split_ratio: float = training_pipeline.DATA_TRAINNER_TRAIN_TEST_SPLIT_RATION
        # Threshold for overfitting/underfitting detection
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD
