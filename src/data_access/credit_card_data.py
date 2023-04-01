import sys
from typing import Optional

import numpy as np
import pandas as pd
import json
from src.configuration.mongo_db_connection import MongoDBClient
from src.constant.database import DATABASE_NAME
from src.exception import CustomException


class CreditCardData:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        """
        Creates connection with Mongo DB
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)

        except Exception as e:
            raise CustomException(e, sys)

    def save_csv_file(self, transaction_file_path, identity_file_path, collection_name: str, database_name: Optional[str] = None):
        """
        Save Dataframe into Mongo DB
        """
        try:
            transection_data_frame=pd.read_csv(transaction_file_path, nrows=50000)
            identity_data_frame=pd.read_csv(identity_file_path, nrows=50000)
            data_frame = transection_data_frame.merge(identity_data_frame, how='left', left_index=True, right_index=True)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise CustomException(e, sys)


    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise CustomException(e, sys)