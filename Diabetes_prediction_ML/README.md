# Diabetes Prediction using Machine Learning

## Overview

This project demonstrates a machine learning approach to predicting diabetes using the PIMA Diabetes Dataset. The project includes data collection, preprocessing, model training, and evaluation, as well as a system to make predictions on new data inputs.

## Dependencies

To run this script, you need the following Python packages:
- numpy
- pandas
- scikit-learn

You can install these packages using pip:

```sh
pip install numpy pandas scikit-learn
```

## Dataset

The dataset used is the PIMA Diabetes Dataset. It is loaded into a pandas DataFrame and includes the following features:

- `Pregnancies`
- `Glucose`
- `BloodPressure`
- `SkinThickness`
- `Insulin`
- `BMI`
- `DiabetesPedigreeFunction`
- `Age`
- `Outcome` (Target variable: 0 for non-diabetic, 1 for diabetic)

## Steps

### 1. Importing the Dependencies

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
```

### 2. Data Collection and Analysis

Load the dataset:

```python
diabetes_dataset = pd.read_csv('/path/to/diabetes.csv')
```

### 3. Data Preprocessing

#### Statistical Summary

```python
diabetes_dataset.describe()
```

#### Data Standardization

```python
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
```

### 4. Splitting the Data

Split the data into training and testing sets:

```python
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
```

### 5. Training the Model

Train a Support Vector Machine (SVM) classifier:

```python
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)
```

### 6. Model Evaluation

Evaluate the model using accuracy score:

```python
training_data_accuracy = accuracy_score(classifier.predict(X_train), Y_train)
test_data_accuracy = accuracy_score(classifier.predict(X_test), Y_test)
```

### 7. Making Predictions

To make a prediction for new input data:

```python
input_data = (5,116,74,0,0,25.6,0.201,30)
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
std_data = scaler.transform(input_data_reshaped)
prediction = classifier.predict(std_data)

if (prediction[0] == 0):
    print('The person is not diabetic')
else:
    print('The person is diabetic')
```

## How to Run

1. Ensure all dependencies are installed.
2. Load your dataset into the specified path.
3. Run the script using a Python interpreter.

```sh
python diabetes_prediction_ML.py
```

## Results

The accuracy score for the training and test datasets will be printed in the console. Additionally, you can input new data to predict whether a person is diabetic or not.

## Notes

- Ensure the correct path to the dataset is provided.
- The script is designed to work in a Python environment and has been tested using Google Colab.
- Customize the input data for making new predictions as needed.

For any issues or contributions, feel free to open an issue or submit a pull request.