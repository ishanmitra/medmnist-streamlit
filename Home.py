from openai import OpenAI
import streamlit as st

st.set_page_config(
    page_title="Progressive Healthcare | 2024",
    page_icon="💡",
)

st.write("# 🦾Progressive Health Care Imaging through Multimodal Classification")

# st.sidebar.success("Select a demo above.")

st.markdown(
    """
    An opportunistic approach to deploying next-gen medical applications in the field of in-depth medical analysis through multimodal deep learning models.
    """
)

"Click on the **bold text** or use the navigation links in the sidebar."
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("![](app/static/path.png)")
    st.page_link("pages/8_Colon_Pathology.py", label="**Colon Pathology**")

    st.markdown("![](app/static/OCT.png)")
    st.page_link("pages/6_OCT_for_Glaucoma_and_Diabetic_Retinopathy.py", label="**OCT for Diabetic Retinopathy**")

    st.markdown("![](app/static/breast.png)")
    st.page_link("pages/2_Breast_Ultrasound_for_Cancer_Detection.py", label="**Breast Cancer Ultrasound**")

with col2:  
    st.markdown("![](app/static/chest.png)")
    st.page_link("pages/4_Chest_XRay_for_Multiple_Diagnosis.py", label="**Chest XRay for Multiple Diagnosis**")

    st.markdown("![](app/static/pneumonia.png)")
    st.page_link("pages/1_Chest_XRay_for_Pneumonia_Detection.py", label="**XRay for Pneumonia Detection**")

    st.markdown("![](app/static/blood.png)")
    st.page_link("pages/7_Blood_Cell_Check_under_Microscope.py", label="**Blood Cell under Microscope**")

with col3:  
    st.markdown("![](app/static/derma.png)")
    st.page_link("pages/5_Dermatoscope_for_Skin_Diagnosis.py", label="**Dermatoscope for Skin Diagnosis**")

    st.markdown("![](app/static/retina.png)")
    st.page_link("pages/3_Fundus_Retina_Severity_Checkup.py", label="**Fundus Retina Severity Checkup**")

    st.markdown("![](app/static/tissue.png)")
    st.page_link("pages/9_Kidney_Cortex_under_Microscope.py", label="**Kidney Cortex under Microscope**")

st.markdown(
    """
    ## Acknowledgement and Citations
    *The database which made this ML application possible is the [MedMNIST database](https://arxiv.org/abs/2110.14795)
    The image data and its labels were indispensable towards the development of this web application.*
    """
)

with st.sidebar:
    st.title("MedMNIST GPT")

    api = st.text_input("Enter OpenAI API Key", type="password")

    if api:
        client = OpenAI(api_key=api)

    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-3.5-turbo"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask me about MedMNIST!"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            stream = client.chat.completions.create(
                model=st.session_state["openai_model"],
                messages=[
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages
                ],
                stream=True,
            )
            response = st.write_stream(stream)
        st.session_state.messages.append({"role": "assistant", "content": response})