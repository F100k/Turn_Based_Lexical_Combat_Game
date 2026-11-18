from glove_loader import load_glove_model, get_cosine_similarity
# นำเข้าฟังก์ชัน Soft Domain Gate ที่เราเพิ่งทำกันไป
# (สมมติว่าคุณรวมไว้ใน glove_loader.py หรือไฟล์จัดการเกม)

def test_lexical_combat_domain():
    """
    ทดสอบระบบตรวจจับโดเมน (Domain Gate) และความเหมือนกับคำเป้าหมาย (Target Word)
    """
    glove_file = "glove.6B.100d.txt"
    embeddings = load_glove_model(glove_file)
    print("-" * 50)
    
    # จำลองสถานการณ์ด่านในเกม (ตามแนวทาง 4 โดเมน)
    test_stages = [
        {"domain": "Business", "target": "executive"},
        {"domain": "Technology", "target": "algorithm"},
        {"domain": "Science", "target": "catalyst"}
    ]
    
    # คำที่ผู้เล่นทดลองพิมพ์เข้ามา
    player_guesses = ["ceo", "manager", "banana", "computer"]
    
    for stage in test_stages:
        print(f"\n=== Stage Domain: [{stage['domain'].upper()}] | Target: '{stage['target']}' ===")
        
        for guess in player_guesses:
            sim = get_cosine_similarity(guess, stage['target'], embeddings)
            print(f"  * Player typed: '{guess}' -> Cosine Similarity with target: {sim:.4f}")

if __name__ == "__main__":
    test_lexical_combat_domain()