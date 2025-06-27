# preprocessing/text_cleaner.py
import re

def clean_arabic_text(text):
    text = text.strip()
    text = re.sub(r'[^\u0600-\u06FF\s]', '', text)  # Keep Arabic letters and space
    text = re.sub(r'\s+', ' ', text)
    return text
