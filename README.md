# Credit Risk Predictions Models 
This is a Jupyter Notebook that looks at historial data to predict high risk loans. Some of the imputs are loan size, debt to income, interest rate, borrower income, and total debt. From these preamiters we train a model to assign a 1 or 0 to find if this is a high risk loan or not. This was made using different sklearn modles and our own code.
---

## Technologies

This project leverages python 3.7 with the following packages:

* [pandas](https://github.com/pandas-dev/pandas) - For the command line interface, help page, and entrypoint.

* [numbpy](https://github.com/numpy/numpy) - The fundamental package for scientific computing with Python

* [metaplot](https://github.com/matplotlib/matplotlib) - For entrypoint and help page.

* [imblean.metrics](http://glemaitre.github.io/imbalanced-learn/generated/imblearn.metrics.classification_report_imbalanced.html) - Build a classification report based on metrics used with imbalanced dataset

* [sklean.metrics](https://github.com/scikit-learn/scikit-learn) - Simple and efficient tools for predictive data analysis

---

## Installation Guide

Before running the application first install the following dependencies. Note that if you are running on the cloud and not locally you will have to run all lines of code.

```
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
```


---

## Usage

Acitvate a Jupyter Lab Notebook by having the kernal installed and typing `jupyter lab` in your terminal. 

---

## Examples
```
# Instantiate the Logistic Regression model
# Assign a random_state parameter of 1 to the model
lr_model_resamp = LogisticRegression(random_state=1)

# Fit the model using the resampled training data
lr_model_resamp.fit(X_resampled, y_resampled)

# Make a prediction using the testing data
y_resamp_pred = lr_model_resamp.predict(X_test)
```

---

## Contributors

DU Starter Code
Terrence McCoy


---

## License

MIT
