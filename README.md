# 📝 Note Summarizer  

A web-based application that generates concise summaries of long text documents. Built with **Streamlit** and transformer-based NLP models, it helps students, researchers, and professionals quickly extract key points from lengthy notes, articles, or reports.  

---

## Features  
- Interactive web app powered by **Streamlit**  
- Summarizes long text into adjustable lengths (Very Small → Large)  
- Uses pre-trained transformer models from **Hugging Face Transformers**  
- Simple and clean UI for quick usage  
- Extensible for future features (e.g., question generation from text)  

---

## 🛠 Tech Stack  
- **Python**  
- **Streamlit** (UI framework)  
- **Hugging Face Transformers** (NLP models)  
- **PyTorch** (depending on model backend)  

---

## 📂 Project Structure  
```
Summarizer/
│── main.py                # Main Streamlit application
│── requirements.txt      # Project dependencies
│── README.md             # Project documentation
```

---

## ⚡ Installation & Usage  

1. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:  
   ```bash
   streamlit run app.py
   ```

3. Open the app in your browser at `http://localhost:8501/`.  

---

---

## 📜 License  
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  

---

## 🙌 Acknowledgements  
- [Hugging Face Transformers](https://huggingface.co/transformers/)  
- [Streamlit](https://streamlit.io/)  
- Pre-trained model  **BART** for summarization  
