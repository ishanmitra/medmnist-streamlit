import streamlit as st
import pandas as pd
import random

from medmnist import RetinaMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_retinaMNIST():
	  return RetinaMNIST(split="test", download=True, size=224)

retina_test_dataset = load_retinaMNIST()

'''# Fundus Retina Severity Checkup'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\retinamnist_test_[AUC]0.724_[ACC]0.475@resnet18_224_3.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    st.write(f"**Severity:** {max_column_id - 1}")

    retina_test_dataset[ROW_NO - 1][0]

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    # class_df.index = ['pneumonia', 'normal']
    class_df

    st.bar_chart(class_df)