from src.utils.main_utils import load_numpy_array_data
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import DataTransformationArtifact, ModelTrainerArtifact
from src.entity.config_entity import ModelTrainerConfig
import os,sys
from lightgbm import LGBMClassifier
from src.ml.metric.classification_metric import get_classification_score
from src.ml.model.estimator import CreditCardModel
from src.constant import training_pipeline
from src.utils.main_utils import save_object,load_object, read_yaml_file
from sklearn.model_selection import train_test_split, GridSearchCV


class ModelTrainer:
    """
    Class for training a machine learning model.

    Attributes:
        model_trainer_config (ModelTrainerConfig): Configuration for the model trainer.
        data_transformation_artifact (DataTransformationArtifact): Data transformation artifact containing
            preprocessed data and other transformation information.
    """
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        """
        Initialize ModelTrainer object.

        Args:
            model_trainer_config (ModelTrainerConfig): Configuration for the model trainer.
            data_transformation_artifact (DataTransformationArtifact): Data transformation artifact containing
                preprocessed data and other transformation information.
        """
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            self._schema_config = read_yaml_file(training_pipeline.SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException(e, sys)


    def perform_hyper_paramter_tunig(self, x_train, y_train):
        # Define the estimator
        lgbmclassifier = LGBMClassifier(random_state=training_pipeline.RANDOM_SEED)

        # Define the parameters gird
        param_grid = self._schema_config['param_grid']

        # run grid search
        grid = GridSearchCV(lgbmclassifier, param_grid=param_grid, refit = True, verbose = 3, n_jobs=-1,cv = 3)
        
        # fit the model for grid search 
        grid.fit(x_train, y_train)

        lgbmclassifier = grid.best_estimator_

        return lgbmclassifier

    

    def train_model(self, x_train, y_train):
        """
        Train a machine learning model using the provided training data.

        Args:
            x_train (numpy array or DataFrame): Input features for training.
            y_train (numpy array or Series): Target labels for training.

        Returns:
            model (object): Trained machine learning model.
        """
        try:
            xgb_clf = XGBClassifier()
            xgb_clf.fit(x_train, y_train)
            return xgb_clf
        except Exception as e:
            raise CustomException(e, sys)
    
    
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        """
        Initiates the training process for the model using the data transformation artifacts and model trainer configuration.

        Returns:
            ModelTrainerArtifact: An object containing the trained model file path and train/test classification metrics.
        """
        try:
            # Load transformed data
            train_file_path = self.data_transformation_artifact.transformed_train_data_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_data_file_path

            X = load_numpy_array_data(train_file_path)
            y = load_numpy_array_data(test_file_path)
            

            x_train, x_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.model_trainer_config.train_test_split_ratio, 
                random_state=training_pipeline.RANDOM_SEED
            )

            # Train the model
            model = self.perform_hyper_paramter_tunig(x_train, y_train)
            y_train_pred = model.predict(x_train)
            classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)

            if classification_train_metric.f1_score <= self.model_trainer_config.expected_accuracy:
                raise Exception("Trained model is not good enough to provide expected accuracy")

            y_test_pred = model.predict(x_test)
            classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)

            # Check for overfitting and underfitting
            diff = abs(classification_train_metric.f1_score - classification_test_metric.f1_score)

            if diff > self.model_trainer_config.overfitting_underfitting_threshold:
                raise Exception("Model is not good. Try to do more experimentation.")

            #preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)

            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path, exist_ok=True)
            #model = CreditCardModel(preprocessor=preprocessor, model=model)
            save_object(self.model_trainer_config.trained_model_file_path, obj=model)

            # Create model trainer artifact
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=classification_train_metric,
                test_metric_artifact=classification_test_metric
            )
            logging.info(f"Model trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)
