# Import libraries
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np

st.title('📖 Exam Performance App')

st.info('Type in your **Study Hours** and **Previous Exam Score** to see if you will pass or fail the Current test. ')

# loading the dataset
df = pd.read_csv("dataset.csv")

# divide dataset into X and y
# X -> featureset
X = df.drop('Pass/Fail', axis=1)  # Features

# y -> target label
y = df['Pass/Fail']  # Target variable

with st.expander("**Data**", expanded=False, icon=None):
    st.text("raw dataset")
    df
    st.text("feature set")
    X
    st.text("target label")
    y

# data is already in Numerical form where: pass = 1 and fail = 0
passing_students = df[df['Pass/Fail'] == 1]
failing_students = df[df['Pass/Fail'] == 0]

# Simple data visualization
with st.expander("**Visualization**", expanded=False):
    st.text("Scatter Plot")
    st.scatter_chart(data=df, x='Study Hours', y='Previous Exam Score', x_label="Study Hours", y_label="Previous Exam Score", color='Pass/Fail', use_container_width=True)
    # create 2 columns to show off the two categories of students
    col1, col2= st.columns(2)
    # print data for passing and failing students separatly
    with col1:
        st.text("Passing Students")
        passing_students
    with col2:
        st.text("Failing Students")
        failing_students

# calculate min value to pass for both marks and Hours
min_hours=passing_students['Study Hours'].min()
min_marks=passing_students['Previous Exam Score'].min()

# some key insights
with st.expander("**Key Insights**", expanded=False):
    st.info('the minimum no. of hours needed to study to Pass:')
    st.success(min_hours)
    st.divider()
    st.info('the minimum no. of Marks needed to Score in prev test to Pass:')
    st.success(min_marks)

clf = RandomForestClassifier()
clf.fit(X,y)
predict = clf.predict(X) # returns predicted class labels
predict_propability = clf.predict_proba(X)
predict_propability
