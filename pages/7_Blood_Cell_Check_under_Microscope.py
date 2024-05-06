import streamlit as st
import pandas as pd
import random
from PIL import Image

from medmnist import BloodMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_bloodMNIST():
	  return BloodMNIST(split="test", download=True)

blood_test_dataset = load_bloodMNIST()

'''# Blood Cell Check under Microscope'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\bloodmnist_test_[AUC]0.998_[ACC]0.960@resnet18_28_2.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    blood_class = ['basophil','eosinophil','erythroblast','immature granulocytes(myelocytes, \
    metamyelocytes and promyelocytes)', 'lymphocyte','monocyte','neutrophil','platelet']

    st.write(f"**Classification:** {blood_class[max_column_id - 1].title()}")

    st.image(blood_test_dataset[ROW_NO - 1][0].resize((224, 224), Image.LANCZOS))

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['neutrophil', 'eosinophil',
    'immature granulocytes(myelocytes metamyelocytes and promyelocytes)',
    'platelet','erythroblast','monocyte','basophil','lymphocyte']
    class_df

    st.bar_chart(class_df)