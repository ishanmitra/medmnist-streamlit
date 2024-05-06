import streamlit as st
import pandas as pd
import random

from medmnist import PneumoniaMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_pneumoniaMNIST():
	  return PneumoniaMNIST(split="test", download=True, size=224)

pneumonia_test_dataset = load_pneumoniaMNIST()

'''# Chest XRay for Pneumonia Detection'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\pneumoniamnist_test_[AUC]0.964_[ACC]0.867@resnet18_224_2.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    path_class = ['normal', 'pneumonia']

    st.write(f"**Classification:** {path_class[max_column_id - 1].title()}")

    pneumonia_test_dataset[ROW_NO - 1][0]

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['pneumonia', 'normal']
    class_df

    st.bar_chart(class_df)