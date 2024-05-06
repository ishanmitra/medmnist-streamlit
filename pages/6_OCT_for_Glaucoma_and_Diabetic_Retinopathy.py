import streamlit as st
import pandas as pd
import random
from PIL import Image

from medmnist import OCTMNIST

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

@st.cache_data
def load_octMNIST():
	  return OCTMNIST(split="test", download=True)

oct_test_dataset = load_octMNIST()

'''# OCT for Glaucoma and Diabetic Retinopathy'''

col1, col2 = st.columns(2)

with col1:
    def load_data(url):
        df = pd.read_csv(url, header=None)
        df = df.drop(0, axis=1)
        
        return df

    df = load_data("predictions\\octmnist_test_[AUC]0.946_[ACC]0.755@resnet18_28_2.csv")

    ROW_NO = random.randint(0, len(df)-1)
    "### Report Index: " + str(ROW_NO)

    max_column_id = df.iloc[ROW_NO].idxmax()

    oct_class = ['choroidal neovascularization','diabetic macular edema','drusen','normal']
   
    st.write(f"**Classification:** {oct_class[max_column_id - 1].title()}")

    st.image(oct_test_dataset[ROW_NO - 1][0].resize((224, 224), Image.LANCZOS))

    if st.button("Classify another report", type="primary"):
        ROW_NO = random.randint(0, len(df)-1)

with col2:
    """
    ### Prediction on Test Reports
    """

    class_df = df.idxmax(axis=1).value_counts().to_frame()
    class_df.index = ['choroidal neovascularization','normal','diabetic macular edema','drusen']
    class_df

    st.bar_chart(class_df)