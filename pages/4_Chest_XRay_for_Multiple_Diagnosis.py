import streamlit as st
import pandas as pd
import random
from PIL import Image

from medmnist import ChestMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_chestMNIST():
	  return ChestMNIST(split="test", download=True)

chest_test_dataset = load_chestMNIST()

'''# Chest XRay for Multiple Diagnosis'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\chestmnist_test_[AUC]0.772_[ACC]0.948@resnet18_28_3.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    chest_class = ['atelectasis','cardiomegaly','effusion','infiltration','mass','nodule',
    'pneumonia','pneumothorax','consolidation','edema','emphysema','fibrosis','pleural','hernia']

    st.write(f"**Classification:** {chest_class[max_column_id - 1].title()}")

    st.image(chest_test_dataset[ROW_NO - 1][0].resize((224, 224), Image.LANCZOS))

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['mass','infiltration','cardiomegaly','consolidation','nodule',
    'pneumonia','effusion','fibrosis','hernia','edema','emphysema','pleural']
    class_df

    st.bar_chart(class_df)