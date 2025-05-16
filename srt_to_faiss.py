import os
import faiss
import pysrt
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer

from typing import List, Tuple

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def detect_languages(text_lines: List[str]) -> Tuple[str, str]:
    """
    Split subtitle lines into English and Russian based on language detection.
    """
    en_lines, ru_lines = [], []
    for line in text_lines:
        try:
            lang = detect(line)
            if lang == 'ru':
                ru_lines.append(line)
            else:
                en_lines.append(line)
        except:
            continue  # skip unknown language lines
    return "\n".join(en_lines), "\n".join(ru_lines)

def parse_srt_file(filepath: str):
    subs = pysrt.open(filepath, encoding='utf-8')
    entries = []
    for sub in subs:
        lines = sub.text.strip().split('\n')
        en_text, ru_text = detect_languages(lines)
        combined = f"{en_text} {ru_text}".strip()
        entries.append({
            "start": sub.start.to_time(),
            "end": sub.end.to_time(),
            "en": en_text,
            "ru": ru_text,
            "text": combined,
            "source": os.path.basename(filepath)
        })
    return entries

def embed_and_index(entries: List[dict]):
    texts = [e["text"] for e in entries]
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    return index, entries

def process_folder(folder_path: str):
    all_entries = []
    for file in os.listdir(folder_path):
        if file.endswith('.srt'):
            filepath = os.path.join(folder_path, file)
            print(f"Processing: {file}")
            entries = parse_srt_file(filepath)
            all_entries.extend(entries)

    index, metadata = embed_and_index(all_entries)
    return index, metadata

def search_index(index, metadata, query: str, k: int = 5):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, k)
    results = []
    for idx, dist in zip(indices[0], distances[0]):
        entry = metadata[idx]
        results.append({
            "score": float(dist),
            "source": entry["source"],
            "start": entry["start"],
            "end": entry["end"],
            "en": entry["en"],
            "ru": entry["ru"]
        })
    return results

if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "./subs"
    index, metadata = process_folder(folder)

    while True:
        query = input("\nSearch: ").strip()
        if query.lower() in ["exit", "quit"]: break
        results = search_index(index, metadata, query)
        for r in results:
            print(f"[{r['start']} - {r['end']}] {r['source']}")
            print(f"EN: {r['en']}")
            print(f"RU: {r['ru']}")
            print("---")
