import streamlit as st
import pandas as pd
import random
from PIL import Image

from medmnist import DermaMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_dermaMNIST():
	  return DermaMNIST(split="test", download=True)

derma_test_dataset = load_dermaMNIST()

'''# Dermatoscope for Skin Diagnosis'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\dermamnist_test_[AUC]0.917_[ACC]0.741@resnet18_28_3.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    derma_class = ['actinic keratoses and intraepithelial carcinoma','basal cell carcinoma',
    'benign keratosis-like lesions','dermatofibroma','melanoma','melanocytic nevi','vascular lesions']

    st.write(f"**Classification:** {derma_class[max_column_id - 1].title()}")

    st.image(derma_test_dataset[ROW_NO - 1][0].resize((224, 224), Image.LANCZOS))

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['melanocytic nevi','benign keratosis-like lesions','melanoma',
      'basal cell carcinoma','actinic keratoses and intraepithelial carcinoma','vascular lesions','dermatofibroma']
    class_df

    st.bar_chart(class_df)