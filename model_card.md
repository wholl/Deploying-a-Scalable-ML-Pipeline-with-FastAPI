# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This project uses a Random Forest classifier from scikit-learn to predict whether a person's income is above or below $50K per year using census data. The project includes data preprocessing, model training, prediction, saving the model, loading the model, FastAPI endpoints, and model slice evaluation.

## Intended Use

This model was created as part of the Udacity and WGU Machine Learning DevOps project. It demonstrates how to build and deploy a machine learning pipeline using Python, FastAPI, and automated testing tools. The predictions should not be used for real decisions.

## Training Data

The model was trained using the Census Income dataset. The dataset includes features such as age, education, occupation, marital status, race, sex, workclass, and hours worked per week. Categorical features were converted using one-hot encoding before training the model.

The data was retrieved from: https://archive.ics.uci.edu/dataset/20/census+income

## Evaluation Data

The original dataset was split into training and testing datasets using a train-test split. About 20% of the data was reserved for testing and the remaining data was used for training.

## Metrics

Precision: 0.7601
Recall: 0.6224
F1: 0.6844

## Ethical Considerations

Because the dataset contains demographic information such as race and sex, the model may reflect biases that exist in the original data. Predictions from this model could unfairly impact some groups if used in real-world situations. Additional fairness testing would be needed before using this model in production.

## Caveats and Recommendations

This project was developed for learning purposes and is not intended for real-world use. The model could likely be improved with additional tuning, testing, and monitoring. Further evaluation would also be recommended before using the model with real users or business decisions.
