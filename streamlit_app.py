# Simple ML app
# By Chanin Nantasenamat (Data Professor)
# https://youtube.com/dataprofessor

# Importing requisite libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Title of the app
st.title('🎈 Simple ML App')

# Load dataset
df = pd.read_csv('https://raw.githubusercontent.com/dataprofessor/data/master/iris.csv')

# Separate to X and y
X = df.drop('Species', axis=1)
y = df.Species

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model building
rf = RandomForestClassifier(max_depth=2, max_features=4, n_estimators=100, random_state=0)

# Apply model to make predictions
y_train_pred = rf.predict(X_train)
y_test_pred = rf.predict(X_test)

# Model performance
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Print data overview
st.subheader('Data overview')
st.write(f'Number of samples (rows): `{df.shape[0]}`')
st.write(f'Number of variables (columns): `{df.shape[1]}`')

# Print model performance
st.subheader('Model performance')
st.write(f'Accuracy (Training set): `{train_accuracy}`')
st.write(f'Accuracy (Test set): `{test_accuracy}`')
