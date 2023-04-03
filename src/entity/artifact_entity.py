from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    ingested_file_path: str


@dataclass
class DataValidationArtifact:
#    validation_status: bool
    valid_file_path: str
    invalid_file_path: str
    

@dataclass
class DataPreprationArtifact:
    validation_status: bool
    prepared_train_file_path: str
    prepared_test_file_path: str
    drift_report_file_path: str
