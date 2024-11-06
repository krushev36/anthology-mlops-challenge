# Random Survival Forest Model to predict Churn on Netflix Subscribers

This repository contains the implementation of a Random Survival Forest model, located in `mlops/models/NetflixModel.py`. The Random Survival Forest is a non-parametric ensemble method used for survival analysis. It extends the random forest algorithm to handle censored data, making it suitable for predicting time-to-event outcomes.

## Key Features

- **Handling Censored Data**: Effectively manages censored data, which is common in survival analysis.
- **Ensemble Learning**: Utilizes multiple decision trees to improve prediction accuracy and robustness.
- **Non-Parametric**: Does not assume a specific distribution for the survival times, providing flexibility in modeling.

## Usage

To use the Random Survival Forest model, import the `NetflixModel` class from the `mlops/models/NetflixModel.py` file and follow the provided documentation to fit the model to your data and make predictions.

```python
from mlops.models.NetflixModel import NetflixModel

# Initialize the model
model = NetflixModel()

# Fit the model to your data
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

For more detailed instructions and examples, please refer to the documentation within the `NetflixModel.py` file.