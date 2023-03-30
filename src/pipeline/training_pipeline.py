from src.entity.config_entity import TrainingPipelineConfig
from src.exception import CustomException
import sys,os
from src.logger import logging

class TrainPipeline:
    is_pipeline_running=False
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_data_validaton(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_data_transformation(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)
    
    def start_model_trainer(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_model_evaluation(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)

    def start_model_pusher(self):
        try:
           pass
        except  Exception as e:
            raise  CustomException(e,sys)


    def run_pipeline(self):
        try:
            pass
        except  Exception as e:
            raise  CustomException(e,sys)