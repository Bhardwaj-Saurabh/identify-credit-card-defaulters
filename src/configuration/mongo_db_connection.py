import pymongo
from src.constant.database import DATABASE_NAME
from src.constant.env_variable import MONGODB_URL_KEY
from src.exception import CustomException
import certifi
import os, sys
from src.logger import logging

ca = certifi.where()

class MongoDBClient:
    """
    Class to establish a connection with MongoDB and interact with the database.
    """
    # Class-level attribute to store the MongoDB client instance
    client = None
    def __init__(self, database_name=DATABASE_NAME) -> None:
        """
        Initializes the MongoDB client and establishes a connection with the specified database.

        Args:
            database_name (str): Name of the MongoDB database to connect to.
        """
        try:
            # Check if client instance already exists, if not, create a new one
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                logging.info(f"Connecting to MongoDB at {mongo_db_url}")
                if "localhost" in mongo_db_url:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url)
                else:
                    MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            self.client = MongoDBClient.client
            self.database = self.client[database_name]
            self.database_name = database_name
            logging.info(f"Connected to MongoDB database: {database_name}")
        except Exception as e:
            raise CustomException(e, sys)

