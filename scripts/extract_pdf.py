from pypdf import PdfReader

reader = PdfReader("2512.24880v1.pdf")
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("mhc_paper.txt", "w") as f:
    f.write(text)
    
print("Text extracted to mhc_paper.txt")
