import streamlit as st
import re
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

def chunk_text(text, max_tokens=900, overlap=50):  # max_tokens for safety margin
   
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0

    while start < len(tokens):
        # Calculate end position
        end = min(start + max_tokens, len(tokens))
        
        # Extract chunk tokens
        chunk_tokens = tokens[start:end]
        
        # Decode to text
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        # Verify the chunk doesn't exceed limits when re-tokenized
        # (Sometimes decoding and re-encoding can change token count slightly)
        verification_tokens = tokenizer.encode(chunk_text, add_special_tokens=True)
        if len(verification_tokens) > 1024:
            # If it exceeds, reduce the chunk size
            reduced_end = start + max_tokens - 100  # Reduce by 100 tokens
            chunk_tokens = tokens[start:reduced_end]
            chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        
        chunks.append(chunk_text)
        
        # Move start position with overlap
        start = end - overlap
        
        # Ensure we don't get stuck in infinite loop
        if start >= end - overlap:
            start = end

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
    size_option = st.session_state["summary_length"]

    size_map = {
        "Very Small": 0.1,    
        "Small": 0.2,          
        "Medium": 0.3,       
        "Large": 0.4         
    }

    size_ratio = size_map[size_option]
    text = clean_text(st.session_state.input_text_value)

    total_tokens = len(tokenizer.encode(text, add_special_tokens=True))

    if total_tokens <= 1024:
        # Text is small enough to summarize directly
        max_len = min(int(total_tokens * size_ratio), 142)
        min_len = max(10, int(max_len * 0.5))
        
        try:
            summary = model(
                text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False
            )[0]['summary_text']
            st.session_state.summarized_text = summary
        except Exception as e:
            st.error(f"Error during summarization: {e}")
            st.session_state.summarized_text = ""
    else:
        # Text needs to be chunked
        chunks = chunk_text(text)
        final_summaries = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Calculate token count for this chunk
                num_tokens = len(tokenizer.encode(chunk, add_special_tokens=True))
                
                # Skip if chunk is too large (shouldn't happen with our chunking, but safety check)
                if num_tokens > 1024:
                    continue
                
                # Calculate appropriate summary lengths based on compression ratio
                max_len = min(int(num_tokens * size_ratio), 142)
                min_len = max(10, int(max_len * 0.5))
                
                summary = model(chunk, max_length=max_len, min_length=min_len, do_sample=False)
                final_summaries.append(summary[0]['summary_text'])
                
            except Exception as e:
                st.error(f"Error processing chunk {i+1}: {e}")
                continue
        
        # Join all summaries
        st.session_state.summarized_text = " ".join(final_summaries)
    
    


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

    /* Streamlit menu & footer */
    .css-9ycgxx, footer {
        display: none !important;
    }

    /* Gap between columns */
    .css-1r6slb0 {
        gap: 2rem !important;
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

    st.caption(f"No. of words = {len(summarized_text.split())}")





