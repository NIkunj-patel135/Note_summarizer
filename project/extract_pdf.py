import pdfplumber


with pdfplumber.open("Exercise2_PDF.pdf") as pdf:
    text = ""
    for page in pdf.pages:
        text += page.extract_text() + "\n"

print(text)
