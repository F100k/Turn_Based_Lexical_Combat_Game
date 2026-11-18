from glove_loader import load_glove_model, calculate_dynamic_damage

if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    
    # 1. โหลดโมเดล GloVe
    embeddings = load_glove_model(glove_file)
    print("-" * 50)
    
    # สมมติว่ามอนสเตอร์ตัวนี้มี Target Word คือ "executive" (หรือคำสั่งโจมตี "attack")
    target = "executive"
    boss_hp = 200 # เลือดมอนสเตอร์
    
    print(f"Target Word (Boss Weakness/Target): '{target}' (Boss HP: {boss_hp})\n")
    
    # จำลองคำที่ผู้เล่นลองพิมพ์โจมตีเข้ามาด้วยความหมายต่างๆ
    player_guesses = ["ceo", "director", "manager", "employee", "banana", "guitar"]
    
    for guess in player_guesses:
        result = calculate_dynamic_damage(guess, target, embeddings, max_hp=boss_hp)
        print(f"Player typed: '{guess}'")
        print(f" -> Similarity: {result['similarity']}")
        print(f" -> Combat Tier: {result['tier']}")
        print(f" -> Damage Dealt: -{result['damage']} HP")
        print("-" * 40)