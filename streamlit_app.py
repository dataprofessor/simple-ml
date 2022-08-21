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


# Input widgets
st.sidebar.subheader('Model parameters')
st_test_size = st.sidebar.slider('Size of test set (test_size)', 0.1, 0.9, 0.2)
st_max_features = st.sidebar.slider('Maximum number of features (max_features)', 1, 4, 4)
st_n_estimators = st.sidebar.slider('Number of estimators (n_estimators)', 100, 1000, 200)


# Separate to X and y
X = df.drop('Species', axis=1)
y = df.Species

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=st_test_size, random_state=42)

# Model building
rf = RandomForestClassifier(max_depth=2, max_features=st_max_features, n_estimators=st_n_estimators, random_state=42)
rf.fit(X_train, y_train)

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
st.write(f'Variable names: `{list(df.columns)}`')

df_species = df.Species.value_counts()
st.write(df_species)
#st.bar_chart

# Print model performance
st.subheader('Model performance')
st.write(f'Accuracy (Training set): `{train_accuracy}`')
st.write(f'Accuracy (Test set): `{test_accuracy}`')
