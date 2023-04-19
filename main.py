from src.configuration.mongo_db_connection import MongoDBClient
from src.exception import CustomException
from src.pipeline.training_pipeline import TrainPipeline 
import os, sys
from src.utils.main_utils import read_yaml_file
from src.logger import logging


env_file_path=os.path.join(os.getcwd(),"env.yaml")

def set_env_variable(env_file_path):
    if os.getenv('MONGO_DB_URL',None) is None:
        env_config = read_yaml_file(env_file_path)
        os.environ['MONGO_DB_URL']=env_config['MONGO_DB_URL']


def main():
    try:
        set_env_variable(env_file_path)
        training_pipeline = TrainPipeline()
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

if __name__=="__main__":
    main()