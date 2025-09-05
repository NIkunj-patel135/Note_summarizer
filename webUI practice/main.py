import streamlit as st
# from transformers import pipeline
# import torch

# device = 0 if torch.cuda.is_available() else -1

# summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", device=device)



if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

if "input_text_value" not in st.session_state:
    st.session_state.input_text_value = "" 

if "summarized_text" not in st.session_state:
    st.session_state.summarized_text = ""


def update_summarized_text():
    st.session_state.summarized_text = st.session_state.input_text_value

st.markdown(
    """
    <style>
    ::-webkit-scrollbar {
        display: none;
    }
    html, body, [class*="block-container"] {
        overflow: hidden !important;
    }
    .typing {
        font-size: 32px;
        font-weight: bold;
        color: white;
        border-right: 3px solid orange; /* cursor */
        white-space: nowrap;
        overflow: hidden;
        display: inline-block;
        animation: typing 2s steps(11, end) forwards, blink 0.7s step-end infinite;
        
    }

    @keyframes typing {
        from { width: 0ch }
        to { width: 11ch } /* match number of characters */
    }

    @keyframes blink {
        50% { border-color: transparent }
    }
    </style>

    <h1 class="typing">Summarizer!</h1>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    input_text = st.text_area(
    "Enter your notes:",
    height=300,  # starting height
    placeholder="Type here...",
    key="input_text_value",
    )
    st.caption(f"No. of words = {len(input_text)}")
    summarize_button = st.button(
        "Summarize",
        type="primary",
        on_click=update_summarized_text
    )

with col2:
    summerized_text = st.text_area(
    "Summarized Text:",
    height=300,  # starting height
    placeholder="Summarized Text",
    value=st.session_state.summarized_text,
    key="summarized_notes",
    )
    st.caption(f"No. of words = {len(st.session_state.summarized_text)}")


# summarizer(input_text, max_length=60, min_length=40, do_sample=False) 