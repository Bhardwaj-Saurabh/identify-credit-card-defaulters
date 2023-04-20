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
    This class helps to export entire MongoDB record as a pandas DataFrame.
    """
    def __init__(self):
        """
        Creates a connection with MongoDB.
        """
        try:
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
        except Exception as e:
            raise CustomException(e, sys)

    def save_csv_file(
        self,
        transaction_file_path: str,
        identity_file_path: str,
        collection_name: str,
        database_name: Optional[str] = None
    ) -> int:
        """
        Save Dataframe into MongoDB.

        Args:
        -----------
        transaction_file_path : str
            File path of the transaction data CSV file.
        identity_file_path : str
            File path of the identity data CSV file.
        collection_name : str
            Name of the MongoDB collection to save the data.
        database_name : Optional[str], default=None
            Name of the MongoDB database. If not provided, the default database is used.

        Returns:
        -----------
        int
            Number of records saved to MongoDB.

        Raises:
        -----------
        CustomException
            If any error occurs during data saving.
        """
        try:
            transection_data_frame = pd.read_csv(transaction_file_path, nrows=50000)
            identity_data_frame = pd.read_csv(identity_file_path, nrows=50000)
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
        self, collection_name: str, database_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Export entire MongoDB collection as a pandas DataFrame.

        Args:
        -----------
        collection_name : str
            Name of the MongoDB collection to export.
        database_name : Optional[str], default=None
            Name of the MongoDB database. If not provided, the default database is used.

        Returns:
        -----------
        pd.DataFrame
            DataFrame containing the data from the MongoDB collection.

        Raises:
        -----------
        CustomException
            If any error occurs during data export.
        """
        try:
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
