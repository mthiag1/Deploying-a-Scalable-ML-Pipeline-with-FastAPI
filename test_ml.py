import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from ml.model import train_model, compute_model_metrics, inference
from ml.data import process_data

# Import the original data and use a subset as sample data for processing and testing
file = "data/census.csv"
df = pd.read_csv(file)
sample_data = df.head(20)

# Split the sample data into training and test data
train, test = train_test_split(sample_data, test_size=0.20, random_state=42)

# Features to be tested on
test_features = [ "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]

# Process data for testing
X_train, y_train, encoder, lb = process_data(train,
    categorical_features=test_features, label= "salary", training=True
)

X_test, y_test, _, _ = process_data(test,
    categorical_features=test_features, label= "salary", training=False,
    encoder=encoder, lb=lb
)

# implement the first test. 
def test_model():
    """
    Train the model with training data and then test to ensure the model is a RandomForest Classifier
    INPUT: None
    OUTPUT: None
    Returns: None

    """

    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier)


# implement the second test. 

def test_metrics():
    """
    Train and predict the model. Calculate the metrics. Verify the data type of the returned metrics matches float data type.
    INPUT: None
    OUTPUT: None
    Returns: None

    """
    model = train_model(X_train, y_train)
    preds = inference(model, X_test)
    Precision, Recall, F1 = compute_model_metrics(y_test, preds)
    assert isinstance(Precision, float)
    assert isinstance(Recall, float)
    assert isinstance(F1, float)


# TODO: implement the third test. Change the function name and input as needed
def test_data_size():
    """
    Test whether the test data size is 20% of the data
    INPUT: None
    OUTPUT: None
    Returns: None

    """
    assert len(test)/len(sample_data) == 0.20
