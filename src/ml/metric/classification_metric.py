from src.entity.artifact_entity import ClassificationMetricArtifact
from src.exception import CustomException
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    """
    Function to calculate classification metric scores.

    Args:
        y_true (array-like): Ground truth labels.
        y_pred (array-like): Predicted labels.

    Returns:
        ClassificationMetricArtifact: Object containing classification metric scores.
    """
    try:
        # Calculate classification metric scores
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        # Create ClassificationMetricArtifact object
        classification_metric = ClassificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )
        return classification_metric
    except Exception as e:
        raise CustomException(e, sys)
