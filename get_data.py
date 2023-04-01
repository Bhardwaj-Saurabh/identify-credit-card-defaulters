file_path="/config/workspace/aps_failure_training_set1.csv"
from src.data_access.credit_card_data import CreditCardData
from src.constant.database import DATABASE_NAME
from src.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from main import set_env_variable
import os
if __name__=='__main__':
    transaction_data_file_path="/Data/train_transaction.csv"
    identity_data_file_path="/Data/train_identity.csv"
    env_file_path='/env.yaml'
    set_env_variable(env_file_path)
    sd = CreditCardData()
    print(sd.mongo_client.database)
    if DATA_INGESTION_COLLECTION_NAME in sd.mongo_client.database.list_collection_names():
        sd.mongo_client.database[DATA_INGESTION_COLLECTION_NAME].drop()
    sd.save_csv_file(transaction_data_file_path, identity_data_file_path, collection_name=DATA_INGESTION_COLLECTION_NAME)
