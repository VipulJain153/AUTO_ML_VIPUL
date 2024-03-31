import streamlit as st
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import profile_report
import pandas as pd,os
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.metrics import mean_squared_error,accuracy_score
import pickle as pk,joblib
from sklearn.datasets import fetch_california_housing
from process import preprocess

california_housing = fetch_california_housing(as_frame=True)
path = "source_data.csv"
task = None

with st.sidebar:
    st.header("Navigation")
    radios = ["Model Configurations", "Data Analysis and Preprocessing", "Train Best Model"]
    radio = st.radio("Select Process",radios)

if radio==radios[0]:
    st.header("Vipul Automatic Machine Learning")

    tasks = ["Select Task", "Regression", "Classification", "Clustering", "Association", "Object Detection", \
              "Object Tracking", "Image Segmentation", "Distance Measurement", "Body Detection"]
    task = st.selectbox("Type",tasks)
    data = st.file_uploader("Upload Data")

    if data!=None:
        df = pd.read_csv(data)
        st.dataframe(df)
        df.to_csv(path,index=False)

if os.path.exists(path) and task!="Select Task":    
    
    df = pd.read_csv(path)
    # df = california_housing.data
    # print(df)

    if radio==radios[1]:
        if task!="Clustering" and task!="Association":
            labels = st.multiselect("Target Varible(s)", list(df.columns))
            if labels:
                    with open("labels.pk","wb") as f:
                        pk.dump(labels,f)
        # st_profile_report(profile_report.ProfileReport(df))

    elif radio==radios[2]:
        if os.path.exists("labels.pk"):
            with open("labels.pk","rb") as f:
                target = pk.load(f)[0]
            X,y = preprocess(df,target)
            if task=="Regression":
                rnf = RandomForestRegressor(n_estimators=500)
                rnf.fit(X,y)
                joblib.dump(rnf, "model.joblib")
                st.download_button("Download Model", "model.joblib", file_name="model.joblib")
                msg =f'Model\'s MSE:{mean_squared_error(y,rnf.predict(X))}'
                st.success(body=msg,icon='😍')
            else:
                rnf = RandomForestClassifier(n_estimators=500)
                rnf.fit(X,y)
                joblib.dump(rnf, "model.joblib")
                st.download_button("Download Model", "model.joblib", file_name="model.joblib")
                msg =f'Model\'s Accuracy:{accuracy_score(y,rnf.predict(X))*100}%'
                st.success(body=msg,icon='😍')

else:
    st.warning("Please Define Task and DataFrame")