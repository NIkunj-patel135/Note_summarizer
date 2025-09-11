import streamlit as st
from transformers import pipeline
import torch
import pdfplumber
import time

@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="philschmid/bart-large-cnn-samsum", device=device)

model = load_summarizer()

def extract_pdf():
    uploaded_file = st.session_state["uploaded_file"]
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        st.session_state.input_text_value = text

st.set_page_config(layout="wide")



if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "input_text_value" not in st.session_state:
    st.session_state.input_text_value = ""

if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""

def update_summarized_text():

    st.session_state.summarized_text = model(st.session_state.input_text_value,max_length=300,min_length=100,do_sample=False)[0]['summary_text']

st.markdown(
    """
    <style>
    /* Hide scrollbars */
    ::-webkit-scrollbar {
        display: none;
    }
    html, body, [class*="block-container"] {
        overflow: hidden !important;
    }
    
    /* Center content with max width and padding */
    .block-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 2rem 3rem !important;
    }
        
    /* Typing animation */
    .typing {
        font-size: 50px !important;
        font-weight: bold;
        color: white;
        border-right: 3px solid orange;
        white-space: nowrap;
        overflow: hidden;
        display: inline-block;
        animation: typing 2s steps(11, end) forwards, blink 0.7s step-end infinite;
        margin-bottom: 2rem;
    }
    
    @keyframes typing {
        from { width: 0ch }
        to { width: 11ch }
    }
    
    @keyframes blink {
        50% { border-color: transparent }
    }
    
    /* Hide Streamlit menu */
    .css-9ycgxx {
        display: none;
    }
    
    /* Add some spacing between columns */
    .css-1r6slb0 {
        gap: 2rem;
    }
    </style>
    <div style="text-align: center;">
        <h1 class="typing">Summarizer!</h1>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area(
        "Enter your Text Here",
        height=400,
        placeholder="Type here...",
        key="input_text_value",
        label_visibility="collapsed"
    )
    
    in_col2, in_col3 = st.columns(2)
    with in_col2:
        summarize_button = st.button(
            "Summarize",
            type="primary",
            on_click=update_summarized_text
        )
        st.caption(f"No. of words = {len(input_text)}")
    with in_col3:
        uploaded_file = st.file_uploader(
            "Upload PDF",
            label_visibility="collapsed",
            type="pdf",
            key="uploaded_file", 
            on_change=extract_pdf,
        )



with col2:
    summarized_text = st.text_area(
        "Summarized Text",
        height=400,
        placeholder="Summarized Text",
        value=st.session_state.summarized_text,
        key="summarized_notes",
        label_visibility="collapsed"
    )
    st.slider("Summarzied Text Length",min_value=50,max_value=300,step=50)
    st.caption(f"No. of words = {len(summarized_text)}")




