import os

SAVED_MODEL_DIR =os.path.join("saved_models")


# defining common constant variable for training pipeline
TARGET_COLUMN = "isFraud"
PIPELINE_NAME: str = "creditcard"
ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "creditcarddata.csv"

TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"

PREPROCSSING_OBJECT_FILE_NAME = "preprocessing.pkl"
MODEL_FILE_NAME = "model.pkl"
SCHEMA_FILE_PATH = os.path.join("config", "schema.yaml")
SCHEMA_DROP_COLS = "drop_columns"


"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "creditcard"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"


"""
Data Validation realted contant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"

"""
Data prepration realted contant start with DATA_PREPRATION VAR NAME
"""
DATA_PREPARATION_DIR_NAME: str = "data_prepration"
DATA_PREPARATION_TRAIN_TEST_SPLIT_RATION: float = 0.2
DATA_PREPARATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_PREPARATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"

"""
Data Transformation ralated constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
