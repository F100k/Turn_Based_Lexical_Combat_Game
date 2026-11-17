import numpy as np

def load_glove_model(glove_file_path):
    """
    function to load GloVe .txt file directly into a Python Dictionary
    using only NumPy
    """
    print(f"Loading GloVe file from {glove_file_path}...")
    embeddings = {}
    
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            word = parts[0]
            # แปลงค่าตัวเลขเวกเตอร์ที่เหลือให้เป็น NumPy Array (Float32)
            vector = np.array(parts[1:], dtype=np.float32)
            embeddings[word] = vector
            
    print(f"all words: {len(embeddings):,} in system vocabulary")
    return embeddings

def get_cosine_similarity(word1, word2, embeddings):
    """
    Cosine Similarity between 2 words
    """
    if word1 not in embeddings or word2 not in embeddings:
        return 0.0  # ถ้าคำใดคำหนึ่งไม่มีในโมเดล ให้คืนค่า 0
    
    v1 = embeddings[word1]
    v2 = embeddings[word2]
    
    # คำนวณตามสูตร Cosine Similarity: (A • B) / (||A|| * ||B||)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    
    if norm_v1 == 0 or norm_v2 == 0:
        return 0.0
        
    return float(dot_product / (norm_v1 * norm_v2))

if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    
    try:
        # 1. โหลดโมเดล
        glove_embeddings = load_glove_model(glove_file)
        
        # 2. ทดสอบวัดความเหมือนของคำ
        w1, w2 = "king", "queen"
        sim = get_cosine_similarity(w1, w2, glove_embeddings)
        print(f"Similarity '{w1}' and '{w2}': {sim:.4f}")
        
        w3, w4 = "king", "apple"
        sim2 = get_cosine_similarity(w3, w4, glove_embeddings)
        print(f"Similarity '{w3}' and '{w4}': {sim2:.4f}")
        
    except FileNotFoundError:
        print(f"\n[!] not found '{glove_file}' in this folder")
        print("please download the .txt file of GloVe and place it in the glove_module folder before running again")