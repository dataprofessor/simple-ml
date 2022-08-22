# Simple ML app
# By Chanin Nantasenamat (Data Professor)
# https://youtube.com/dataprofessor

# Importing requisite libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# Page configuration
st.set_page_config(
     page_title='Simple ML App',
     page_icon='🎈',
     layout='wide',
     initial_sidebar_state='expanded')

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

# Data overview
st.subheader('Data overview')
st.write(f'Number of samples (rows): `{df.shape[0]}`')
st.write(f'Number of variables (columns): `{df.shape[1]}`')
st.write(f'Variable names: `{list(df.columns)}`')

# EDA
st.subheader('EDA')

data_describe = df.describe()
st.write(data_describe)

data_sepal_length = pd.qcut(df['Sepal.Length'], 3, labels=["low", "medium", "high"]).value_counts()
data_sepal_width = pd.qcut(df['Sepal.Width'], 3, labels=["low", "medium", "high"]).value_counts()
data_petal_length = pd.qcut(df['Petal.Length'], 3, labels=["low", "medium", "high"]).value_counts()
data_petal_width = pd.qcut(df['Petal.Width'], 3, labels=["low", "medium", "high"]).value_counts()

data_species = df.Species.value_counts()

eda_col1, eda_col2, eda_col3, eda_col4, eda_col5 =  st.columns(5)
with eda_col1:
  st.bar_chart(data_sepal_length)
with eda_col2:
  st.bar_chart(data_sepal_width)
with eda_col3:
  st.bar_chart(data_petal_length)
with eda_col4:
  st.bar_chart(data_petal_width)
with eda_col5:
  st.bar_chart(data_species)

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

# Cross-validation
rf = RandomForestClassifier(max_depth=2, max_features=st_max_features, n_estimators=st_n_estimators, random_state=42)
cv_scores = cross_val_score(rf, X_train, y_train, cv=5).mean()

# Model performance
st.subheader('Model performance')

model_col1, model_col2, model_col3 = st.columns(3)
with model_col1:
  st.metric('Training set', round(train_accuracy,3), '')
with model_col2:
  st.metric('Test set', test_accuracy, '')
with model_col3:
  st.metric('5-fold CV set', cv_scores)
