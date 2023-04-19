from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    ingested_file_path: str


@dataclass
class DataValidationArtifact:
    valid_file_path: str
    invalid_file_path: str
    

@dataclass
class DataPreprationArtifact:
    prepared_data_file_path: str
    drift_report_file_path: str

@dataclass
class DataTransformationArtifact:
    transformed_data_file_path: str
