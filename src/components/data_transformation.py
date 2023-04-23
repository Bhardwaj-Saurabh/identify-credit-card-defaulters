import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataPreparationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.constant.training_pipeline import SCHEMA_FILE_PATH, SCHEMA_DROP_COLS, SCHEMA_PCA_COLS
from src.utils.main_utils import read_yaml_file
from src.utils.main_utils import save_numpy_array_data, reduce_mem_usage
from src.ml.preprocessor.preprocess_data import perform_PCA, missing_values_and_scaling_encoder, frequency_encoder


class DataTransformation:
    """
    Class to represent data transformation for the training pipeline.
    """

    def __init__(self, data_preparation_artifact: DataPreparationArtifact, 
                 data_transformation_config: DataTransformationConfig):
        """
        Initializes the data transformation process.

        Args:
            data_prepration_artifact (DataPreprationArtifact): Output reference of the data preparation artifact stage.
            data_transformation_config (DataTransformationConfig): Configuration for the data transformation process.
        """
        try:
            self.data_preparation_artifact = data_preparation_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        """
        Reads data from a file and returns a pandas DataFrame.

        Args:
            file_path (str): File path of the data file to be read.

        Returns:
            pd.DataFrame: Pandas DataFrame containing the data.
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        """
        Initiates the data transformation process.

        Returns:
            DataTransformationArtifact: Data transformation artifact containing the transformed data file path.
        """
        try:
            # Read prepared data file
            df = DataTransformation.read_data(self.data_preparation_artifact.prepared_data_file_path)

            # Filter columns for data transformation
            selected_columns = [col for col in df.columns if col.startswith('V')]
            
            filter_col = self._schema_config[SCHEMA_PCA_COLS]
            filter_col = [col for col in filter_col if col in selected_columns]
        
            
            # Perform missing values imputation and scaling using encoder
            df = missing_values_and_scaling_encoder(df, filter_col)

            # Perform PCA
            df = perform_PCA(df, filter_col, prefix='PCA_V_', n_components=30)

            # Perform frequency encoding
            df = frequency_encoder(df)

            # drop unnecessary columns
            #df.drop(self._schema_config[SCHEMA_DROP_COLS])

            # Reduce memory usage
            df = reduce_mem_usage(df)

            # Extract input features and target feature
            X = df.drop(columns=[TARGET_COLUMN], axis=1)
            y = df[TARGET_COLUMN]

            for col in X.columns:
                logging.info(str(X[col].isna().sum()))

            # replace inf
            X = X.replace([np.inf, -np.inf], np.nan)

            # Perform resampling using SMOTE-Tomek
            #smt = SMOTETomek(sampling_strategy="minority")
            #input_feature_final, target_feature_final = smt.fit_resample(input_feature_df, target_feature)

            # Save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_data_file_path, array=X)
            save_numpy_array_data(self.data_transformation_config.transformed_test_data_file_path, array=y)
            
            # Prepare data transformation artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_data_file_path=self.data_transformation_config.transformed_data_file_path,
                transformed_train_data_file_path=self.data_transformation_config.transformed_train_data_file_path,
                transformed_test_data_file_path=self.data_transformation_config.transformed_test_data_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

