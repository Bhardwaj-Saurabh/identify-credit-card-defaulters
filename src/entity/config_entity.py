
from datetime import datetime
import os
from src.constant  import training_pipeline

class TrainingPipelineConfig:

    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name: str = training_pipeline.PIPELINE_NAME
        self.artifact_dir: str = os.path.join(training_pipeline.ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
        def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
            )
            self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
            )
            self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join( training_pipeline_config.artifact_dir, training_pipeline.DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir: str = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR)
        self.valid_file_path: str = os.path.join(self.valid_data_dir, training_pipeline.FILE_NAME)
        self.invalid_file_path: str = os.path.join(self.invalid_data_dir, training_pipeline.FILE_NAME)

        
class DataPreprationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_prepration_dir: str = os.path.join( training_pipeline_config.artifact_dir, training_pipeline.DATA_PREPRATION_DIR_NAME)
        self.prepared_train_data_file_path: str = os.path.join(self.data_prepration_dir, training_pipeline.TRAIN_FILE_NAME)
        self.prepared_test_data_file_path: str = os.path.join(self.data_prepration_dir, training_pipeline.TEST_FILE_NAME)
        self.prepared_train_test_split_ratio: float = training_pipeline.DATA_PREPRATION_TRAIN_TEST_SPLIT_RATION
        self.drift_report_file_path: str = os.path.join(
            self.data_prepration_dir,
            training_pipeline.DATA_PREPRATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_PREPRATION_DRIFT_REPORT_FILE_NAME,
        )