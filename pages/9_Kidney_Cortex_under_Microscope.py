import streamlit as st
import pandas as pd
import random
from PIL import Image

from medmnist import TissueMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_tissueMNIST():
    return TissueMNIST(split="test", download=True)

tissue_test_dataset = load_tissueMNIST()

'''# Kidney Cortex under Microscope'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\tissuemnist_test_[AUC]0.931_[ACC]0.683@resnet18_28_2.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    tissue_class = ['Collecting Duct, Connecting Tubule','Distal Convoluted Tubule',
    'Glomerular endothelial cells','Interstitial endothelial cells','Leukocytes',
    'Podocytes','Proximal Tubule Segments','Thick Ascending Limb']
    

    st.write(f"**Classification:** {tissue_class[max_column_id - 1].title()}")

    st.image(tissue_test_dataset[ROW_NO - 1][0].resize((224, 224), Image.LANCZOS))

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['Collecting Duct, Connecting Tubule','Proximal Tubule Segments','Thick Ascending Limb',
      'Interstitial endothelial cells','Leukocytes','Glomerular endothelial cells','Podocytes','Distal Convoluted Tubule']
    class_df

    st.bar_chart(class_df)