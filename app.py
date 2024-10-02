import streamlit as st
import pandas as pd
import os
#import ydata_profiling
#from pandas_profiling import ProfileReport
#import pandas_profiling #noqa

#from streamlit_pandas_profiling import st_profile_report
from pycaret.classification import setup, compare_models, pull, save_model

#creating sidebar
with st.sidebar:
    st.image("Gray and Black Simple Studio Logo.png")
    st.title("Automatic ML-Model Report Generator ")
    choice = st.radio("Navigation", ["Upload","ML","Download"])
    st.info("This application will help you to diagnose the dataset and result in report")

if os.path.exists("sourcedata.csv"):
    df=pd.read_csv("sourcedata.csv", index_col=None)

if choice =="Upload":
    st.title("Data Modelling Begins here")
    file = st.file_uploader("Upload Your Dataset")
    if file:
        df=pd.read_csv(file, index_col=None)
        df.to_csv("sourcedata.csv", index=None)
        st.dataframe(df)


# if choice =="Profiling":
#     st.title("Profile Report")
#     profile_report = df.profile_report()
#     st_profile_report(profile_report)

if choice =="ML":
    st.title("Machine Learning Analysis")
    target= st.selectbox("Select the Targeted Column", df.columns)
    if st.button("Train Model"):
        setup(df, target=target)
        setup_df = pull()
        st.info("Desingated ML Model Settings")
        st.dataframe(setup_df)
        best_model = compare_models()
        compare_df=pull()
        st.info("ML Model")
        st.dataframe(compare_df)
        best_model
        save_model(best_model, 'best_model')

if choice == "Download":
    with open("best_model.pkl", 'rb') as f:
        st.download_button("Download the ML Report", f, "trained_model_pkl")