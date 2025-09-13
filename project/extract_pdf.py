# import pdfplumber
# import threading
# lock = threading.Lock()
# with lock:
#     with pdfplumber.open("TFCS_Questions.pdf") as pdf:
#         text = ""
#         for page in pdf.pages:
#             text += page.extract_text() + "\n"

# print(text)

import fitz  # PyMuPDF
# import threading
# lock = threading.Lock()
# with lock:
with fitz.open("TFCS_Questions.pdf") as pdf:
    text = ""
    for page in pdf:
        text += page.get_text("text") + "\n"

print(text)
