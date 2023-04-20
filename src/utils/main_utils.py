import yaml
from src.exception import CustomException
from src.logger import logging
import os,sys
import numpy as np
import pandas as pd
import dill


def save_numpy_array_data(file_path: str, array: np.array) -> None:
        '''
        Save numpy array data to file.

        Args:
            file_path (str): The file path where the data will be saved.
            array (np.array): The numpy array data to be saved.
        '''
        try:
            logging.info("Entered the save_numpy_array_data method of MainUtils class")
            
            # Create directory if not exists
            dir_path = os.path.dirname(file_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Save numpy array data to file
            with open(file_path, "wb") as file_obj:
                np.save(file_obj, array)
                
        except Exception as e:
            # Raise a custom exception with error details and system information
            raise CustomException(e, sys) from e
        finally:
            logging.info("Exited the save_numpy_array_data method of MainUtils class")


def load_numpy_array_data(file_path: str) -> np.array:
        '''
        Load numpy array data from file.

        Args:
            file_path (str): The file path from which the data will be loaded.

        Returns:
            np.array: The loaded numpy array data.
        '''
        try:
            logging.info("Entered the load_numpy_array_data method of MainUtils class")
            
            # Check if file exists
            if not os.path.exists(file_path):
                raise Exception(f"The file: {file_path} does not exist")
            
            # Load numpy array data from file
            with open(file_path, "rb") as file_obj:
                return np.load(file_obj)
            
        except Exception as e:
            # Raise a custom exception with error details and system information
            raise CustomException(e, sys) from e
        finally:
            logging.info("Exited the load_numpy_array_data method of MainUtils class")


def save_object(file_path: str, obj: object) -> None:
        '''
        Save object to file using dill serialization.

        Args:
            file_path (str): The file path where the object will be saved.
            obj (object): The object to be saved.

        Returns:
            None
        '''
        try:
            logging.info("Entered the save_object method of MainUtils class")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Serialize and save object to file
            with open(file_path, "wb") as file_obj:
                dill.dump(obj, file_obj)
            
            logging.info("Exited the save_object method of MainUtils class")
        except Exception as e:
            # Raise a custom exception with error details and system information
            raise CustomException(e, sys) from e


def load_object(file_path: str) -> object:
    '''
    Loads an object from a file using dill library.

    Args:
        file_path (str): The file path from which to load the object.

    Returns:
        object: The loaded object.
    '''
    try:
        logging.info(f"Loading object from file: {file_path}")

        # Check if the file exists
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")

        # Load object from file
        with open(file_path, "rb") as file_obj:
            loaded_object = dill.load(file_obj)

        logging.info(f"Object loaded successfully from file: {file_path}")
        return loaded_object

    except Exception as e:
        # Raise a custom exception with error details and system information
        raise CustomException(e, sys) from e
    
def reduce_mem_usage(df: pd.DataFrame):
    """
    Reduce the memory usage of the dataset by downcasting numeric columns to lower precision types.

    Args:
        df (pd.DataFrame): The input DataFrame to reduce memory usage.

    Returns:
        pd.DataFrame: The DataFrame with reduced memory usage.
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    try:
        # Log the start of memory reduction process
        logging.info("Starting memory reduction of the DataFrame.")
        # Record the initial memory usage
        start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        # Iterate over each column in the DataFrame
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    # Downcast integer columns to lower precision types if possible
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    # Downcast float columns to lower precision types if possible
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
        # Record the final memory usage
        end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
        # Log the completion of memory reduction process
        logging.info("Memory reduction completed.")
        # Log the percentage reduction in memory usage
        logging.info('Memory usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
    except Exception as e:
        # Raise a custom exception with error details and system information
        raise CustomException(e, sys)

def read_yaml_file(file_path: str) -> dict:
    """
    Read and load data from a YAML file.

    Args:
        file_path (str): The file path of the YAML file to read.

    Returns:
        dict: The content of the YAML file as a dictionary.
    """
    try:
        # Log the start of YAML file reading process
        logging.info("Reading YAML file: {}".format(file_path))
        with open(file_path, "rb") as yaml_file:
            # Load the content of the YAML file as a dictionary
            content = yaml.safe_load(yaml_file)
            # Log the completion of YAML file reading process
            logging.info("YAML file reading completed.")
            return content
    except Exception as e:
        # Raise a custom exception with error details and system information
        raise CustomException(e, sys)


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write data to a YAML file.

    Args:
        file_path (str): The file path of the YAML file to write.
        content (object): The data to be written to the YAML file.
        replace (bool, optional): If True, replace the existing file at the file path. Defaults to False.
    """
    try:
        # Log the start of YAML file writing process
        logging.info("Writing YAML file: {}".format(file_path))
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "a") as file:
            # Write the content to the YAML file
            yaml.dump(content, file)
        # Log the completion of YAML file writing process
        logging.info("YAML file writing completed.")
    except Exception as e:
        # Raise a custom exception with error details and system information
        raise CustomException(e, sys)