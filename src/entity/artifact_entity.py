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
