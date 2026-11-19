import os
import time
import requests
from collections import Counter
import pandas as pd
import nltk

# โหลด WordNet จาก NLTK (ใช้เป็นตัวสำรองถ้าหน้า Wikipedia หาไม่เจอ)
for _pkg in ("wordnet", "omw-1.4"):
    nltk.download(_pkg, quiet=True)
from nltk.corpus import wordnet as wn

# 1. กำหนด 4 โดเมนหลักและคำศัพท์ตัวแทน (Business, Technology, Science, Arts)
DOMAINS = ["Business", "Technology", "Science", "Arts"]

DOMAIN_EXAMPLES = {
    "Business": [
        "executive", "ceo", "director", "manager", "revenue", "profit",
        "business logic", "market strategy", "financial analysis",
        "shareholder value", "corporate leadership", "sales forecast"
    ],
    "Technology": [
        "algorithm", "software", "database", "network", "compiler",
        "logic gate", "machine learning", "source code", "cloud computing",
        "data structure", "neural network", "operating system"
    ],
    "Science": [
        "catalyst", "molecule", "quantum mechanics", "chemical reaction",
        "cell biology", "gravity", "atomic structure", "genetic mutation",
        "laboratory experiment", "thermodynamics", "electron", "ecosystem"
    ],
    "Arts": [
        "painting", "sculpture", "abstract painting", "classical music",
        "stage performance", "poetry", "portrait art", "modern dance",
        "cinematography", "watercolor technique", "orchestra", "theater"
    ]
}

CALIBRATION_TARGETS = {
    "executive": "Business", 
    "algorithm": "Technology",
    "catalyst": "Science", 
    "painting": "Arts"
}

GAME_STAGES = [
    {"stage": 1, "enemy": "Corporate Titan", "target": "executive", "domain": "Business", "hp": 180},
    {"stage": 2, "enemy": "Cyber Sentinel", "target": "algorithm", "domain": "Technology", "hp": 200},
    {"stage": 3, "enemy": "Alchemist Golem", "target": "catalyst", "domain": "Science", "hp": 200},
]

# 2. ตั้งค่าโฟลเดอร์ปลายทาง
CORPUS_DIR = "corpus"
os.makedirs(CORPUS_DIR, exist_ok=True)
CORPUS_FILE = os.path.join(CORPUS_DIR, "domain_docs.csv")

WIKI_HEADERS = {"User-Agent": "LexicalCombatGame-TextAnalytics/1.1 (KMUTT CPE393 student project; python-requests)"}
MIN_DOC_WORDS = 12

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = "".join(ch if (ch.isalpha() or ch.isspace()) else " " for ch in text)
    return " ".join(text.split())

def corpus_terms() -> list:
    pairs = []
    for domain in DOMAINS:
        for term in DOMAIN_EXAMPLES.get(domain, []):
            pairs.append((clean_text(term), domain))
    for t, d in CALIBRATION_TARGETS.items():
        pairs.append((clean_text(t), d))
    for s in GAME_STAGES:
        pairs.append((clean_text(s["target"]), s["domain"]))
    
    seen, out = set(), []
    for term, domain in pairs:
        if term and term not in seen:
            seen.add(term)
            out.append((term, domain))
    return out

WIKI_API = "https://en.wikipedia.org/w/api.php"

def _wiki_batch(session, terms: list) -> dict:
    params = {"action": "query", "format": "json", "formatversion": 2, "redirects": 1,
              "prop": "extracts|pageprops", "exintro": 1, "explaintext": 1, "ppprop": "disambiguation",
              "titles": "|".join(terms)}
    try:
        response = session.get(WIKI_API, params=params, timeout=20)
        if response.status_code != 200:
            return {}
        query = response.json().get("query", {})
        alias = {t: t for t in terms}
        for step in ("normalized", "redirects"):
            mapping = {m["from"]: m["to"] for m in query.get(step, [])}
            alias = {t: mapping.get(a, a) for t, a in alias.items()}
        pages = {p["title"]: p for p in query.get("pages", [])}
        out = {}
        for term, title in alias.items():
            page = pages.get(title, {})
            text = " ".join(str(page.get("extract", "")).split())
            if page.get("missing") or "disambiguation" in page.get("pageprops", {}) or len(text.split()) < MIN_DOC_WORDS:
                continue
            out[term] = (title, " ".join(text.split()[:220]))
        return out
    except Exception:
        return {}

def fetch_wikipedia_all(terms: list) -> dict:
    session = requests.Session()
    session.headers.update(WIKI_HEADERS)
    found = {}
    for i in range(0, len(terms), 20):
        found.update(_wiki_batch(session, terms[i:i + 20]))
        time.sleep(0.5)
    return found

def wordnet_text(term: str):
    synsets = wn.synsets(term.replace(" ", "_"), pos=wn.NOUN) or wn.synsets(term.replace(" ", "_"))
    if not synsets and " " in term:
        synsets = wn.synsets(term.split()[-1], pos=wn.NOUN)
    parts = []
    for s in synsets[:3]:
        parts.append(s.definition())
        parts.extend(s.examples()[:2])
    text = ". ".join(parts)
    return (term, text) if text else None

def build_corpus() -> pd.DataFrame:
    terms = corpus_terms()
    found = fetch_wikipedia_all([t for t, _ in terms])
    
    rows, stamp = [], time.strftime("%Y-%m-%d")
    for term, domain in terms:
        w = found.get(term)
        source, hit = ("wikipedia", w) if w else ("wordnet", wordnet_text(term))
        if hit is None:
            continue
        title, text = hit
        rows.append({
            "term": term, "domain": domain, "source": source, "title": title, "text": text,
            "n_words": len(text.split()), "retrieved_at": stamp
        })
    frame = pd.DataFrame(rows)
    frame.insert(0, "doc_id", [f"d{i:03d}" for i in range(len(frame))])
    return frame

print("กำลังสร้าง Domain Document Corpus ให้ตรงกับรูปแบบของกลุ่ม...")
corpus_df = build_corpus()
corpus_df.to_csv(CORPUS_FILE, index=False)
print(f"สำเร็จ! บันทึกไฟล์ลงที่: {CORPUS_FILE} (ทั้งหมด {len(corpus_df)} เอกสาร)")