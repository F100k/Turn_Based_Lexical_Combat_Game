import difflib
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

def test_smart_oov_recovery(oov_list, glove_embeddings, sample_size=5):
    """
    ทดสอบจำลองเหตุการณ์: ถ้าผู้เล่นพิมพ์คำที่เป็น OOV หรือสะกดผิด 
    ระบบจะสามารถกู้ชีพโดยการหาคำที่ใกล้เคียงที่สุดใน GloVe มาแทนได้ไหม
    """
    print("\n--- Testing Smart OOV Recovery (Fallback System) ---")
    
    # ดึงรายชื่อคำทั้งหมดใน GloVe มาเตรียมไว้ให้ difflib ค้นหา
    known_vocab = list(glove_embeddings.keys())
    
    # ลองสุ่มตัวอย่างคำ OOV หรือจำลองคำพิมพ์ผิดของผู้เล่น
    test_samples = oov_list[:sample_size] if len(oov_list) >= sample_size else oov_list
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
    corpus_file = "corpus.txt"
    
    try:
        # โหลดโมเดล GloVe
        glove_embeddings = load_glove_model(glove_file)
        print("-" * 40)
        
        # รันตรวจสอบคำใน Corpus แบบเดิม
        in_vocab, oov_words = check_corpus_vocabulary(corpus_file, glove_embeddings)
        
        # --- เพิ่มส่วนนี้เพื่อให้โชว์ผลการกู้ชีพคำผิดชัดเจน ---
        if oov_words:
            print("\n" + "=" * 40)
            print("DEMO: Testing Smart OOV / Spell Correction:")
            print("=" * 40)
            test_smart_oov_recovery(oov_words, glove_embeddings)
        
    except FileNotFoundError as e:
        print(f"\n[!] Not found: {e}")
        print("Please check if 'glove.6B.100d.txt' and 'corpus.txt' are in the correct folder.")