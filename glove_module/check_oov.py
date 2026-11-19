import difflib
import re
import pandas as pd
from glove_loader import load_glove_model

def check_corpus_vocabulary(corpus_csv_path, glove_embeddings):
    """
    ฟังก์ชันอ่านไฟล์ CSV ของคอร์ปัส (domain_docs.csv) และตรวจสอบคำว่าอยู่ใน GloVe หรือเป็น OOV
    """
    print(f"Reading corpus data from {corpus_csv_path}...")
    
    try:
        corpus_df = pd.read_csv(corpus_csv_path)
    except FileNotFoundError:
        print(f"[!] Error: Not found file '{corpus_csv_path}'")
        return [], []
    
    unique_words = set()
    
    for text in corpus_df['text'].dropna():
        words = re.findall(r'\b[a-zA-Z]+\b', str(text).lower())
        for w in words:
            unique_words.add(w)
                
    print(f"Found unique words in Corpus: {len(unique_words):,} words\n")
    
    in_vocab = []
    oov_words = []
    
    for word in sorted(unique_words):
        if word in glove_embeddings:
            in_vocab.append(word)
        else:
            oov_words.append(word)
            
    total_words = len(unique_words) if len(unique_words) > 0 else 1
    print("=" * 40)
    print(f"Summary of vocabulary check in the corpus:")
    print(f" - Present in GloVe (In-vocab): {len(in_vocab):,} words ({len(in_vocab)/total_words*100:.2f}%)")
    print(f" - Out of Vocabulary (OOV / Not in GloVe): {len(oov_words):,} words ({len(oov_words)/total_words*100:.2f}%)")
    print("=" * 40)
    
    if oov_words:
        print("\nExample (OOV 1st 20 words):")
        print(oov_words[:20])
    
    return in_vocab, oov_words

def test_smart_oov_recovery(oov_list, glove_embeddings, sample_size=5):
    """
    ทดสอบจำลองเหตุการณ์: ถ้าผู้เล่นพิมพ์คำที่เป็น OOV หรือสะกดผิด 
    ระบบจะสามารถกู้ชีพโดยการหาคำที่ใกล้เคียงที่สุดใน GloVe มาแทนได้ไหม
    """
    print("\n--- Testing Smart OOV Recovery (Fallback System) ---")
    
    # ดึงรายชื่อคำทั้งหมดใน GloVe มาเตรียมไว้ให้ difflib ค้นหา
    known_vocab = list(glove_embeddings.keys())
    
    # ลองสุ่มตัวอย่างคำ OOV หรือจำลองคำพิมพ์ผิดของผู้เล่น
    test_samples = oov_list[:sample_size] if len(oov_list) >= sample_size else list(oov_list)
    # เพิ่มคำพิมพ์ผิดจำลองเข้าไปเทสด้วย
    test_samples.extend(["exectiv", "directr", "managr"]) 
    
    for word in test_samples:
        # ใช้ difflib หาคำที่ใกล้เคียงที่สุดใน GloVe (ความเหมือน 60% ขึ้นไป)
        matches = difflib.get_close_matches(word, known_vocab, n=1, cutoff=0.6)
        
        if matches:
            corrected = matches[0]
            print(f"  * Player typed [OOV]: '{word}' -> 🛠️ System corrected to: '{corrected}'")
        else:
            print(f"  * Player typed [OOV]: '{word}' -> ❌ Too unique, cannot recover.")

if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    corpus_file = "corpus/domain_docs.csv"  # เปลี่ยนมาอ้างอิงไฟล์ CSV ของคอร์ปัสใหม่
    
    try:
        # โหลดโมเดล GloVe
        glove_embeddings = load_glove_model(glove_file)
        print("-" * 40)
        
        # รันตรวจสอบคำใน Corpus รูปแบบตาราง CSV
        in_vocab, oov_words = check_corpus_vocabulary(corpus_file, glove_embeddings)
        
        # โชว์ผลการกู้ชีพคำผิด
        if oov_words:
            print("\n" + "=" * 40)
            print("DEMO: Testing Smart OOV / Spell Correction:")
            print("=" * 40)
            test_smart_oov_recovery(oov_words, glove_embeddings)
        
    except FileNotFoundError as e:
        print(f"\n[!] Not found: {e}")
        print("Please check if 'glove.6B.100d.txt' and 'corpus/domain_docs.csv' are in the correct folder.")