import streamlit as st
import re
import pyperclip
from transformers import pipeline, AutoTokenizer
import torch
import pdfplumber
import time


@st.cache_resource
def load_summarizer():
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)

tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
model = load_summarizer()

def chunk(text, max_chars=3500):
    """Simple character-based chunking - no tokenizer overhead"""
    chunks = []
    sentences = text.split('. ')
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk + sentence) < max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def extract_pdf():
    uploaded_file = st.session_state["uploaded_file"]
    if uploaded_file is not None:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        st.session_state.input_text_value = text 

st.set_page_config(layout="wide")

def clean_text(text: str) -> str:
    if not text:
        return ""

    # Normalize spaces (replace non-breaking spaces etc.)
    text = text.replace("\xa0", " ").replace("\u200b", " ")

    # Remove bullet points and special symbols
    text = re.sub(r"[•·●▪◆■●★▶►‐–—−●]", " ", text)

    # Remove multiple hyphens/underscores (page breaks or separators)
    text = re.sub(r"[-_]{2,}", " ", text)

    # Remove extra newlines (convert to space)
    text = re.sub(r"\n+", " ", text)

    # Remove isolated numbers (page numbers, references)
    text = re.sub(r"\b\d+\b", " ", text)

    # Remove weird multiple punctuation (e.g., "!!!", "...", "??")
    text = re.sub(r"([!?.,]){2,}", r"\1", text)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "input_text_value" not in st.session_state:
    st.session_state.input_text_value = ""

if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""



def update_summarized_text():
    text = clean_text(st.session_state.input_text_value)
    
    if not text.strip():
        st.session_state.summarized_text = ""
        return
    
    # Simple length estimation - no tokenizer calls
    estimated_tokens = len(text.split()) * 1.3  # Rough estimation
    size_option = st.session_state["summary_length"]
    
    # Simplified compression ratios
    compression_map = {
        "Very Small": 0.1, 
        "Small": 0.15, 
        "Medium": 0.2, 
        "Large": 0.3
    }
    compression_ratio = compression_map[size_option]
    
    try:
        if estimated_tokens <= 1000:  # Direct summarization
            target_length = max(30, int(len(text.split()) * compression_ratio * 1.3))
            
            summary = model(
                text,
                max_length=min(target_length, 512),
                min_length=max(15, target_length // 2),
                do_sample=False,
                truncation=True
            )[0]['summary_text']
            
            st.session_state.summarized_text = summary
        
        else:  # Simple chunking without re-summarization
            chunks = chunk(text)
            summaries = []
            
            for chunk in chunks:
                chunk_words = len(chunk.split())
                target_length = max(20, int(chunk_words * compression_ratio * 1.3))
                
                summary = model(
                    chunk,
                    max_length=min(target_length, 100),
                    min_length=max(10, target_length // 3),
                    do_sample=False,
                    truncation=True
                )[0]['summary_text']
                
                summaries.append(summary)
            
            # Simple concatenation - no re-summarization
            st.session_state.summarized_text = " ".join(summaries)
    
    except Exception as e:
        st.error(f"Summarization error: {e}")
        st.session_state.summarized_text = ""


def copyFunction():
    pyperclip.copy(st.session_state["summarized_notes"])

st.markdown(
    """
    <style>

        /* General layout */
    html, body, [class*="block-container"] {
        margin: 0;
        padding: 0;
        overflow-x: hidden;
    }

    /* Center content with max width */
    .block-container {
        max-width: 1200px !important;
        margin: 0 auto !important;
        padding: 2rem 3rem !important;
    }

    /* Typing animation */
    .typing {
        font-size: 4rem; /* Responsive base size */
        font-weight: bold;
        color: white;
        border-right: 3px solid orange;
        white-space: nowrap;
        overflow: hidden;
        display: inline-block;
        animation: typing 2s steps(11, end) forwards, blink 0.7s step-end infinite;
        margin-bottom: 2rem;
        text-align: center;
    }

    @keyframes typing {
        from { width: 0ch; }
        to { width: 11ch; }
    }

    @keyframes blink {
        50% { border-color: transparent; }
    }


    /* Scrollable content for small devices */
    @media (max-width: 768px) {
        .main .block-container,
        .stApp,
        .main,
        section.main > div,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="block-container"] {
            overflow-y: auto !important;
            height: auto !important;
            max-height: none !important;
            padding-top: 80px; !important;
        }

        .typing {
            font-size: 3rem; /* scale down for mobile */
        }

        .block-container {
            padding: 1rem 1.5rem !important;
        }
    }

    /* Hide scrollbars (optional) */
    ::-webkit-scrollbar {
        display: none;
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

        st.caption(f"No. of words = {len(input_text.split())}")

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
    st.select_slider(
        "Summarzied Text Length",
        options=["Very Small", "Small", "Medium", "Large"],
        value="Medium",
        help="Adjust the length of the generated summary.",
        key="summary_length",
    )
    in_col1, in_col2 = st.columns([6,1])
    with in_col1:
        st.caption(f"No. of words = {len(summarized_text.split())}")
    with in_col2:
        st.button(
            "Copy",
            type="secondary",
            on_click= copyFunction
        )
    





