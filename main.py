from src.pipeline.training_pipeline import TrainPipeline 
import os
from src.utils.main_utils import read_yaml_file
from src.logger import logging

# Define the file path for the env.yaml file
env_file_path = os.path.join(os.getcwd(), "env.yaml")

# Function to set environment variable
def set_env_variable(env_file_path):
    # Check if MONGO_DB_URL environment variable is already set
    if os.getenv('MONGO_DB_URL', None) is None:
        # Read the MONGO_DB_URL value from env.yaml
        env_config = read_yaml_file(env_file_path)
        # Set the MONGO_DB_URL environment variable
        os.environ['MONGO_DB_URL'] = env_config['MONGO_DB_URL']

# Main function
def main():
    try:
        # Set environment variable
        set_env_variable(env_file_path)
        # Instantiate the training pipeline
        training_pipeline = TrainPipeline()
        # Run the training pipeline
        training_pipeline.run_pipeline()
    except Exception as e:
        print(e)
        logging.exception(e)

# Entry point
if __name__ == "__main__":
    main()