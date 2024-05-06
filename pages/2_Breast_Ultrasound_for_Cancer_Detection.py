import streamlit as st
import pandas as pd
import random

from medmnist import BreastMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_breastMNIST():
	  return BreastMNIST(split="test", download=True, size=224)

breast_test_dataset = load_breastMNIST()

'''# Breast Ultrasound for Cancer Detection'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\breastmnist_test_[AUC]0.913_[ACC]0.846@resnet18_224_1.csv")



    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    path_class = ['malignant', 'normal, benign']


    st.write(f"**Classification:** {path_class[max_column_id - 1].title()}")

    breast_test_dataset[ROW_NO - 1][0]

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['normal, benign', 'malignant']
    class_df

    st.bar_chart(class_df)