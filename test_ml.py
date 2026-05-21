import numpy as np
from sklearn.ensemble import RandomForestClassifier

from ml.data import apply_label
from ml.model import compute_model_metrics, inference, train_model


def test_apply_label_output():
    """Test that labels are converted into readable income categories"""
    assert apply_label(np.array([1])) == ">50K"
    assert apply_label(np.array([0])) == "<=50K"


def test_model_training():
    """Test that the training function returns a Random Forest model"""
    sample_X = np.array([[5, 1], [10, 0], [15, 1], [20, 0]])
    sample_y = np.array([0, 0, 1, 1])

    trained_model = train_model(sample_X, sample_y)

    assert type(trained_model) is RandomForestClassifier


def test_prediction_count():
    """Test that inference returns the right number of predictions"""
    sample_X = np.array([[2, 1], [4, 0], [6, 1], [8, 0]])
    sample_y = np.array([0, 0, 1, 1])

    trained_model = train_model(sample_X, sample_y)
    predictions = inference(trained_model, sample_X)

    assert len(predictions) == len(sample_y)


def test_metric_values():
    """Test that metrics are as expected"""
    actual = np.array([1, 1, 0, 0])
    predicted = np.array([1, 0, 0, 0])

    precision, recall, fbeta = compute_model_metrics(actual, predicted)

    assert round(precision, 2) == 1.00
    assert round(recall, 2) == 0.50
    assert round(fbeta, 2) == 0.67
