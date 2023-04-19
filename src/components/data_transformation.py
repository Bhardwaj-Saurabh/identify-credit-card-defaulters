import sys

import numpy as np
import pandas as pd
from imblearn.combine import SMOTETomek

from src.constant.training_pipeline import TARGET_COLUMN
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataPreprationArtifact,
)
from src.entity.config_entity import DataTransformationConfig
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import save_numpy_array_data, reduce_mem_usage
from src.ml.preprocessor.preprocess_data import perform_PCA, missing_values_and_scalling_encoder, frequency_encoder


class DataTransformation:
    def __init__(self, data_prepration_artifact: DataPreprationArtifact, 
                    data_transformation_config: DataTransformationConfig):
        """
        :param data_prepration_artifact: Output reference of data prepration artifact stage
        :param data_transformation_config: configuration for data transformation
        """
        try:
            self.data_prepration_artifact = data_prepration_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise CustomException(e, sys)


    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    
    def initiate_data_transformation(self,) -> DataTransformationArtifact:
        try:
            df = DataTransformation.read_data(self.data_prepration_artifact.prepared_data_file_path)
            
            filter_col = df.columns[53:392]
            df = missing_values_and_scalling_encoder(df, filter_col)

            df = perform_PCA(df, filter_col, prefix='PCA_V_', n_components=30)

            df = frequency_encoder(df)

            # Reduce memory usage
            df = reduce_mem_usage(df)

            #training dataframe
            input_feature_df = df.drop(columns=[TARGET_COLUMN], axis=1).values
            target_feature = df[TARGET_COLUMN].values

            #smt = SMOTETomek(sampling_strategy="minority")

            #input_feature_final, target_feature_final = smt.fit_resample(
            #     input_feature_df, target_feature
            # )

            train_arr = np.c_[input_feature_df, target_feature]

            #save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_data_file_path, array=train_arr)          
            
            #preparing artifact
            data_transformation_artifact = DataTransformationArtifact(
                transformed_data_file_path=self.data_transformation_config.transformed_data_file_path,
            )
            logging.info(f"Data transformation artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)