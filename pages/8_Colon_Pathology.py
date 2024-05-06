import streamlit as st
import pandas as pd
import random
from PIL import Image

from medmnist import PathMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_pathMNIST():
    return PathMNIST(split="test", download=True)

path_test_dataset = load_pathMNIST()

'''# Colon Pathology'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\pathmnist_test_[AUC]0.990_[ACC]0.910@resnet18_28_3.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    path_class = ['adipose','background','debris','lymphocytes','mucus','smooth muscle','normal colon mucosa','cancer-associated stroma','colorectal adenocarcinoma epithelium']

    st.write(f"**Classification:** {path_class[max_column_id - 1].title()}")

    st.image(path_test_dataset[ROW_NO - 1][0].resize((224, 224), Image.LANCZOS))

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['adipose','colorectal adenocarcinoma epithelium','background',
      'mucus','normal colon mucosa','smooth muscle','lymphocytes','debris','cancer-associated stroma']
    class_df

    st.bar_chart(class_df)