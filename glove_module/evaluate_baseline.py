import re
import numpy as np
from glove_loader import load_glove_model, get_cosine_similarity

def evaluate_semantic_accuracy(embeddings):
    """
    1. Evaluate Command Matching Accuracy (Extended Version)
    """
    print("\n--- [1] Semantic Command Matching Evaluation (Extended) ---")
    
    test_cases = [
        # --- ชุดทดสอบเดิม ---
        ("strike", "attack", 0.5),   # (คำผู้เล่น, คำสั่งหลักที่คาดหวัง, เกณฑ์ผ่านขั้นต่ำ)
        ("escape", "run", 0.5),
        ("potion", "heal", 0.3),
        ("bag", "inventory", 0.4),
        ("explore", "search", 0.4),
        
        # --- ชุดทดสอบใหม่ (เพิ่มความหลากหลาย) ---
        ("slash", "attack", 0.5),      
        ("flee", "run", 0.5),          
        ("medicine", "heal", 0.4),     
        ("backpack", "inventory", 0.5),
        ("look", "search", 0.4),       
        ("jump", "run", 0.3),          
        ("shield", "attack", 0.2)      
    ]
    
    passed = 0
    for player_input, target_cmd, threshold in test_cases:
        sim = get_cosine_similarity(player_input, target_cmd, embeddings)
        status = "✅ PASS" if sim >= threshold else "❌ FAIL"
        if sim >= threshold:
            passed += 1
        print(f"  * '{player_input}' -> '{target_cmd}': Similarity = {sim:.4f} [{status}]")
        
    accuracy = (passed / len(test_cases)) * 100
    print(f"Command Matching Accuracy: {accuracy:.2f}% ({passed}/{len(test_cases)})")

def evaluate_oov_coverage(corpus_path, embeddings):
    """
    2. evaluate OOV (Out-of-Vocabulary) Coverage
    """
    print("\n--- [2] OOV (Out-of-Vocabulary) Coverage Evaluation ---")
    unique_words = set()
    
    try:
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = re.findall(r'\b[a-zA-Z]+\b', line.lower())
                for w in words:
                    unique_words.add(w)
    except FileNotFoundError:
        print(f"  [!] not found {corpus_path} skipping OOV coverage evaluation")
        return

    in_vocab_count = sum(1 for w in unique_words if w in embeddings)
    total_words = len(unique_words)
    coverage_rate = (in_vocab_count / total_words) * 100 if total_words > 0 else 0

    print(f"  * Repeat Corpus: {total_words:,} คำ")
    print(f"  * Have in GloVe (In-vocab): {in_vocab_count:,} คำ")
    print(f"  * Not in GloVe (OOV): {total_words - in_vocab_count:,} คำ")
    print(f"Vocabulary Coverage Rate: {coverage_rate:.2f}%")

def evaluate_word_analogy(embeddings):
    """
    3. Semantic evaluation (Word Analogy / Vector Arithmetic) - Multi-test version
       Equation: vector(w2) - vector(w1) + vector(w3) = vector(?)
    """
    print("\n--- [3] Word Analogy / Relationship Test (Multiple Examples) ---")
    
    # กำหนดชุดทดสอบความสัมพันธ์หลายๆ รูปแบบ
    analogy_test_cases = [
        {
            "w1": "man", "w2": "king", "w3": "woman", "expected": "queen",
            "candidates": ["queen", "prince", "princess", "kingdom", "castle", "apple", "sword"]
        },
        {
            "w1": "paris", "w2": "france", "w3": "tokyo", "expected": "japan",
            "candidates": ["japan", "tokyo", "asia", "china", "germany", "rome", "king"]
        },
        {
            "w1": "brother", "w2": "man", "w3": "sister", "expected": "woman",
            "candidates": ["woman", "girl", "boy", "father", "mother", "queen", "apple"]
        },
        {
            "w1": "small", "w2": "smaller", "w3": "cold", "expected": "colder",
            "candidates": ["colder", "coldest", "warm", "hot", "freezing", "small", "big"]
        }
    ]
    
    passed_cases = 0
    
    for case in analogy_test_cases:
        w1, w2, w3 = case["w1"], case["w2"], case["w3"]
        expected = case["expected"]
        candidates = case["candidates"]
        
        # ตรวจสอบว่าคำทั้งหมดอยู่ในโมเดล GloVe หรือไม่
        all_words = [w1, w2, w3, expected] + candidates
        if all(w in embeddings for w in all_words):
            # คำนวณเวกเตอร์ผลลัพธ์: v_result = v_w2 - v_w1 + v_w3
            v_result = embeddings[w2] - embeddings[w1] + embeddings[w3]
            
            best_word = None
            highest_sim = -1.0
            
            # ค้นหาคำที่ใกล้เคียงที่สุดจากชุดผู้ท้าชิง (Candidates)
            for cand in candidates:
                v_cand = embeddings[cand]
                sim = np.dot(v_result, v_cand) / (np.linalg.norm(v_result) * np.linalg.norm(v_cand))
                if sim > highest_sim:
                    highest_sim = sim
                    best_word = cand
            
            status = "✅ PASS" if best_word == expected else "⚠️ CHECK"
            if best_word == expected:
                passed_cases += 1
                
            print(f"  * Test: '{w2}' - '{w1}' + '{w3}'")
            print(f"    -> Expected: '{expected}' | Model Choice: '{best_word}' (Sim: {highest_sim:.4f}) [{status}]")
        else:
            print(f"  * Skipped test ('{w2}' - '{w1}' + '{w3}') due to missing vocabulary.")
            
    print(f"Analogy Evaluation Accuracy: {(passed_cases / len(analogy_test_cases)) * 100:.2f}% ({passed_cases}/{len(analogy_test_cases)})")

if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    corpus_file = "corpus.txt"
    
    print("==================================================")
    print("Starting GloVe Model Evaluation Process")
    print("==================================================")
    
    try:
        embeddings = load_glove_model(glove_file)
        
        # รันการประเมินทั้ง 3 ด้าน
        evaluate_semantic_accuracy(embeddings)
        evaluate_oov_coverage(corpus_file, embeddings)
        evaluate_word_analogy(embeddings)
        
        print("\n==================================================")
        print("Evaluation completed! Ready to present results to the team.")
        print("==================================================")
        
    except FileNotFoundError:
        print(f"\n[!] not found {glove_file} please make sure it's placed in the folder correctly")