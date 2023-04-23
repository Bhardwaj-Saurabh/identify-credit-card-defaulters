from src.constant.training_pipeline import SAVED_MODEL_DIR, MODEL_FILE_NAME
from src.exception import CustomException
import os, sys


# Write a code to train model and check the accuracy.
class CreditCardModel:
    """
    Class representing a sensor model for prediction.

    Attributes:
        preprocessor (object): Preprocessor object for data transformation.
        model (object): Model object for prediction.
    """

    def __init__(self, preprocessor, model):
        """
        Initialize SensorModel object.

        Args:
            preprocessor (object): Preprocessor object for data transformation.
            model (object): Model object for prediction.
        """
        try:
            self.preprocessor = preprocessor
            self.model = model
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, x):
        """
        Predict the target labels for given input data.

        Args:
            x (array-like): Input data.

        Returns:
            array-like: Predicted target labels.
        """
        try:
            y_hat = self.model.predict(x)
            return y_hat
        except Exception as e:
            raise CustomException(e, sys)
