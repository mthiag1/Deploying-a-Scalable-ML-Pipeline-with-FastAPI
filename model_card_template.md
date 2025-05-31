# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This is a machine learning model using Random Forest Classifier algoritm from the scikit-learn library.

## Intended Use

This model is to be used to predict whether an individual's income level is greater or less than $50,000 based on demographic information.

## Training Data

The data is collected from publically available Census Bureau data. This data includes features such as age, occupation, marital status, sex, salary, education level and more factors.

## Evaluation Data

25% of the original dataset is used for evaluation seperately as test dataset.The features in the test data contains the same features as the training dataset and it is used to assess the model's performance on new data.

## Metrics
The performance of the model was evaluated using the bellow metrics.

- Precision: 0.7382  
- Recall: 0.6353  
- F1: 0.6829
 
## Ethical Considerations

This data is not a true representation of whole population. The predictions using this data may be biased as well.

## Caveats and Recommendations

Regular updates and retraining of model on updated datasets can help maintain the model's accuracy.