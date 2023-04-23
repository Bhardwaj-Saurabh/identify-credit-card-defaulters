from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    """
    Represents the artifact resulting from data ingestion in a training pipeline.
    
    Attributes:
    -----------
    ingested_file_path : str
        File path of the ingested data.
    """
    ingested_file_path: str


@dataclass
class DataValidationArtifact:
    """
    Represents the artifact resulting from data validation in a training pipeline.
    
    Attributes:
    -----------
    valid_file_path : str
        File path of the valid data.
    invalid_file_path : str
        File path of the invalid data.
    """
    valid_file_path: str
    invalid_file_path: str
    

@dataclass
class DataPreparationArtifact:
    """
    Represents the artifact resulting from data preparation in a training pipeline.
    
    Attributes:
    -----------
    prepared_data_file_path : str
        File path of the prepared data.
    drift_report_file_path : str
        File path of the drift report.
    """
    prepared_data_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    """
    Represents the artifact resulting from data transformation in a training pipeline.
    
    Attributes:
    -----------
    transformed_data_file_path : str
        File path of the transformed data.
    """
    transformed_data_file_path: str
    transformed_train_data_file_path: str
    transformed_test_data_file_path: str

@dataclass
class ClassificationMetricArtifact:
    """
    Data class to store classification metric artifact.

    Attributes:
        f1_score (float): F1 score.
        precision_score (float): Precision score.
        recall_score (float): Recall score.
    """ 
    f1_score: float
    precision_score: float
    recall_score: float

@dataclass
class ModelTrainerArtifact:
    """
    Data class to store model trainer artifact.

    Attributes:
        trained_model_file_path (str): File path of the trained model.
        train_metric_artifact (ClassificationMetricArtifact): Train classification metric artifact.
        test_metric_artifact (ClassificationMetricArtifact): Test classification metric artifact.
    """
    trained_model_file_path: str
    train_metric_artifact: ClassificationMetricArtifact
    test_metric_artifact: ClassificationMetricArtifact

