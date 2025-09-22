# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Start the program.
2.Data preprocessing:
3.Cleanse data,handle missing values,encode categorical variables.
4.Model Training:Fit logistic regression model on preprocessed data.
5.Model Evaluation:Assess model performance using metrics like accuracyprecisioon,recall.
6.Prediction: Predict placement status for new student data using trained model.
7.End the program.
```

## Program:
```

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: ARSHIYA M
RegisterNumber: 212224040029 


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load data
data = pd.read_csv("Placement_Data.csv")

# Copy and drop unnecessary columns
data1 = data.copy()
data1 = data1.drop(["sl_no", "salary"], axis=1)

# Encode categorical columns
le = LabelEncoder()
for col in ["gender", "ssc_b", "hsc_b", "hsc_s", "degree_t", "workex", "specialisation", "status"]:
    data1[col] = le.fit_transform(data1[col])

# Split features and target
x = data1.iloc[:, :-1]
y = data1["status"]

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Logistic Regression
lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

# Predictions
y_pred = lr.predict(x_test)

# Accuracy and report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---- Prediction with new input ----
# Make sure the input follows the SAME column order as x.columns
print("Column order:", list(x.columns))

# Example prediction (you must match values correctly!)
sample = [[1, 80, 1, 90, 1, 1, 90, 1, 0, 85, 1, 85]]  # 12 features
print("Prediction:", lr.predict(sample))
```

## Output:
<img width="1373" height="358" alt="Screenshot 2025-09-22 090528" src="https://github.com/user-attachments/assets/6668dd13-34d5-4d0d-9a92-48eeed21d06f" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
