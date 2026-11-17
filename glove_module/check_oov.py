import re
from glove_loader import load_glove_model

def check_corpus_vocabulary(corpus_path, glove_embeddings):
    """
    Function to read corpus.txt and check if words are in GloVe or OOV (Out of Vocabulary)
    """
    print(f"Reading data from {corpus_path}...")
    
    unique_words = set()
    
    # 1. เปิดไฟล์ corpus.txt และดึงคำศัพท์ทั้งหมดออกมา (ทำความสะอาดเบื้องต้นด้วย RegEx)
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            # แปลงเป็นตัวพิมพ์เล็กและดึงเฉพาะคำภาษาอังกฤษ/ตัวอักษร
            words = re.findall(r'\b[a-zA-Z]+\b', line.lower())
            for w in words:
                unique_words.add(w)
                
    print(f"Found unique words in Corpus: {len(unique_words):,} words\n")
    
    # 2. คัดแยกคำที่มี (In-vocab) และคำที่ไม่มี (OOV) ใน GloVe
    in_vocab = []
    oov_words = []
    
    for word in sorted(unique_words):
        if word in glove_embeddings:
            in_vocab.append(word)
        else:
            oov_words.append(word)
            
    # 3. แสดงผลสรุปสถิติ
    print("=" * 40)
    print(f"Summary of vocabulary check in the game:")
    print(f" - Present in GloVe (In-vocab): {len(in_vocab):,} words ({len(in_vocab)/len(unique_words)*100:.2f}%)")
    print(f" - Out of Vocabulary (OOV / Not in GloVe): {len(oov_words):,} words ({len(oov_words)/len(unique_words)*100:.2f}%)")
    print("=" * 40)
    
    # 4. แสดงตัวอย่างคำที่หลุดโผ (OOV) เพื่อให้คุณดูว่ามีคำเฉพาะทางอะไรบ้าง
    if oov_words:
        print("\nExample (OOV 1st 20 words):")
        print(oov_words[:20])
    
    return in_vocab, oov_words

if __name__ == "__main__":
    # กำหนดเส้นทางไฟล์ GloVe และ Corpus ของคุณ
    glove_file = "glove.6B.100d.txt"
    corpus_file = "corpus.txt"
    
    try:
        # โหลดโมเดล GloVe ที่เราทำไว้ก่อนหน้านี้
        glove_embeddings = load_glove_model(glove_file)
        print("-" * 40)
        
        # รันตรวจสอบคำใน Corpus
        in_vocab, oov_words = check_corpus_vocabulary(corpus_file, glove_embeddings)
        
    except FileNotFoundError as e:
        print(f"\n[!] not found: {e}")
        print("check file that glove.6B.100d.txt and corpus.txt in folder glove_module before running again")