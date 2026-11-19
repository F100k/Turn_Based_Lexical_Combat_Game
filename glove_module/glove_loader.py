import numpy as np
import pandas as pd

def load_glove_model(glove_file_path):
    """
    ฟังก์ชันโหลดไฟล์ GloVe .txt โดยตรงเข้า Python Dictionary ด้วย NumPy
    """
    print(f"Loading GloVe file from {glove_file_path}...")
    embeddings = {}
    
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
            
    print(f"all words: {len(embeddings):,} in system vocabulary")
    return embeddings

def load_domain_corpus(csv_path="corpus/domain_docs.csv"):
    """
    ฟังก์ชันสำหรับโหลดคอร์ปัสรูปแบบใหม่ที่เป็นตารางเอกสารรายโดเมน (จากของลูกเจี๊ยบ)
    """
    try:
        corpus_df = pd.read_csv(csv_path)
        print(f"Loaded domain corpus from {csv_path} ({len(corpus_df)} documents)")
        return corpus_df
    except FileNotFoundError:
        print(f"[!] Warning: Not found '{csv_path}'. Please build the corpus first.")
        return None

def get_cosine_similarity(word1, word2, embeddings):
    """
    คำนวณ Cosine Similarity ระหว่าง 2 คำ
    """
    if word1 not in embeddings or word2 not in embeddings:
        return 0.0  
    
    v1 = embeddings[word1]
    v2 = embeddings[word2]
    
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))

def get_phrase_vector(phrase, embeddings, stopwords=None):
    """
    แปลงวลีหรือประโยคให้เป็นเวกเตอร์เดียวด้วย Mean Pooling และ L2 Normalization 
    พร้อมกรองคำเชื่อม (Stopwords) ออก
    """
    if stopwords is None:
        stopwords = {"a", "an", "the", "in", "on", "at", "by", "for", "with", "and", "or"}
        
    words = phrase.lower().split()
    valid_vectors = []
    
    for w in words:
        if w in embeddings and w not in stopwords:
            valid_vectors.append(embeddings[w])
            
    if not valid_vectors:
        return None
        
    # คำนวณค่าเฉลี่ยเวกเตอร์ (Mean Pooling)
    phrase_vector = np.mean(valid_vectors, axis=0)
    
    # ทำ L2 Normalization ให้เวกเตอร์มีความยาวเป็น 1
    norm = np.linalg.norm(phrase_vector)
    if norm > 0:
        phrase_vector = phrase_vector / norm
        
    return phrase_vector

def get_phrase_similarity(phrase1, phrase2, embeddings):
    """
    เปรียบเทียบความเหมือนระหว่างวลี 2 วลี
    """
    v1 = get_phrase_vector(phrase1, embeddings)
    v2 = get_phrase_vector(phrase2, embeddings)
    
    if v1 is None or v2 is None:
        return 0.0
        
    return float(np.dot(v1, v2))

def calculate_dynamic_damage(player_input, target_word, embeddings, max_hp=100, cutoffs=(0.05, 0.15, 0.30, 0.50)):
    """
    คำนวณดาเมจแบบ Dynamic โดยรองรับค่า Cutoffs ที่จูนมาจาก Grid Search
    """
    sim = get_phrase_similarity(player_input, target_word, embeddings)
    t1, t2, t3, t4 = cutoffs
    
    if sim < t1:
        tier = "BLOCKED / MISS"
        damage = 0
    elif sim < t2:
        tier = "WEAK GRAZE"
        damage = int(max_hp * 0.10)
    elif sim < t3:
        tier = "SOLID HIT"
        damage = int(max_hp * 0.20)
    elif sim < t4:
        tier = "STRONG HIT"
        damage = int(max_hp * 0.35)
    else:
        tier = "CRITICAL STRIKE!"
        damage = int(max_hp * 0.50)
        
    return {
        "similarity": round(sim, 4),
        "tier": tier,
        "damage": damage
    }

DOMAIN_ANCHORS = {
    "Business": [
        "Business finance accounting financial reporting quarterly report annual report balance sheet income statement cash flow revenue profit budget audit forecast",
        "Business management leadership executive chief executive officer CEO president director manager supervisor board of directors chairman team organization",
        "Business operations strategy planning human resources payroll hiring project management sales marketing procurement supply chain logistics customer service"
    ],
    "Technology": [
        "Technology computing computer science software development programming source code debugging compiler API application database",
        "Algorithms and data structures array linked list stack queue tree graph hash table heap sorting searching recursion time complexity pseudocode",
        "Artificial intelligence AI machine learning deep learning neural network model training inference natural language processing computer vision"
    ],
    "Science": [
        "Science scientific research physics chemistry biology experiments hypotheses theories measurements natural phenomena",
        "Chemistry studies molecules, atoms, elements, compounds, chemical reactions, chemical bonds, acids, bases and the periodic table",
        "Physics includes mechanics, gravity, electricity, magnetism, quantum mechanics, particles, electrons, forces, energy and motion"
    ],
    "Arts": [
        "Arts visual art painting sculpture drawing illustration design creativity aesthetics artist gallery museum",
        "Painting is a visual art form that uses pigment, color, composition, canvas, brushwork and artistic expression",
        "Music includes melody, harmony, rhythm, songs, composers, orchestras, symphonies, opera, concerts and instruments such as guitar and piano"
    ]
}

def get_domain_vector(domain_name, embeddings):
    """
    สร้างเวกเตอร์ตัวแทนของโดเมนจากค่าเฉลี่ยของประโยคใน DOMAIN_ANCHORS
    """
    if domain_name not in DOMAIN_ANCHORS:
        return None
        
    vectors = []
    for sentence in DOMAIN_ANCHORS[domain_name]:
        vec = get_phrase_vector(sentence, embeddings)
        if vec is not None:
            vectors.append(vec)
            
    if not vectors:
        return None
        
    domain_vec = np.mean(vectors, axis=0)
    norm = np.linalg.norm(domain_vec)
    if norm > 0:
        domain_vec = domain_vec / norm
    return domain_vec

def soft_domain_gate(player_phrase, required_domain, embeddings, threshold=0.3):
    """
    ระบบเช็กโดเมนสำหรับ 4 หมวด (Business, Technology, Science, Arts)
    """
    phrase_vec = get_phrase_vector(player_phrase, embeddings)
    if phrase_vec is None:
        return {"allowed": False, "multiplier": 0.0, "status": "UNKNOWN / OOV"}
        
    domain_scores = {}
    for dom in DOMAIN_ANCHORS.keys():
        dom_vec = get_domain_vector(dom, embeddings)
        if dom_vec is not None:
            sim = float(np.dot(phrase_vec, dom_vec))
            domain_scores[dom] = sim
            
    if not domain_scores:
        return {"allowed": True, "multiplier": 1.0, "status": "NEUTRAL"}
        
    best_domain = max(domain_scores, key=domain_scores.get)
    best_score = domain_scores[best_domain]
    req_score = domain_scores.get(required_domain, 0.0)
    
    if req_score >= threshold and (best_domain == required_domain or (best_score - req_score) < 0.12):
        if best_domain == required_domain and req_score >= 0.30:
            return {"allowed": True, "multiplier": 1.25, "status": f"PRIMARY ({required_domain}) - Bonus!"}
        else:
            return {"allowed": True, "multiplier": 1.0, "status": f"IN-DOMAIN ({required_domain})"}
    else:
        if req_score >= threshold:
            return {"allowed": True, "multiplier": 0.75, "status": f"SECONDARY (Matched {best_domain} instead)"}
        else:
            return {"allowed": False, "multiplier": 0.0, "status": "BLOCKED (Out of Domain)"}
        
if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    
    try:
        glove_embeddings = load_glove_model(glove_file)
        
        # ทดสอบโหลดคอร์ปัสแบบตารางเอกสาร
        corpus_df = load_domain_corpus()
        if corpus_df is not None:
            print(f"Sample corpus documents loaded: {len(corpus_df)} rows")
        
        w1, w2 = "king", "queen"
        sim = get_cosine_similarity(w1, w2, glove_embeddings)
        print(f"Similarity '{w1}' and '{w2}': {sim:.4f}")
        
    except FileNotFoundError:
        print(f"\n[!] not found '{glove_file}' in this folder")