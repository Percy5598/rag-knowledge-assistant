def character_chunker(
    text,
    chunk_size=300,
    overlap=50
):
    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


import re

def sentence_split(text):
    return re.split(r'[.!?]', text)    