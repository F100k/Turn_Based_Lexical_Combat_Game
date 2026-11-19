import numpy as np
import pandas as pd
import itertools
from glove_loader import load_glove_model, get_phrase_vector, soft_domain_gate, get_cosine_similarity

GATE_TEST_CASES = [
    # (Target, Player Guess, Required Domain, Expected Gate Label)
    ("executive", "ceo", "Business", "in_domain"),
    ("executive", "director", "Business", "in_domain"),
    ("executive", "banana", "Business", "out_of_domain"),
    ("executive", "algorithm", "Business", "out_of_domain"),
    
    ("algorithm", "source code", "Technology", "in_domain"),
    ("algorithm", "database", "Technology", "in_domain"),
    ("algorithm", "abstract painting", "Technology", "out_of_domain"),
    ("algorithm", "ceo", "Technology", "out_of_domain"),
    
    ("catalyst", "molecule", "Science", "in_domain"),
    ("catalyst", "chemical reaction", "Science", "in_domain"),
    ("catalyst", "spreadsheet", "Science", "out_of_domain"),
    ("catalyst", "guitar", "Science", "out_of_domain"),
    
    ("painting", "sculpture", "Arts", "in_domain"),
    ("painting", "orchestra", "Arts", "in_domain"),
    ("painting", "algorithm", "Arts", "out_of_domain"),
    ("painting", "banana", "Arts", "out_of_domain"),
]

SIMULATION_STAGES = [
    {
        "stage": 1, "enemy": "Corporate Titan", "target": "executive", "domain": "Business", "hp": 180, 
        "guesses": ["banana", "employee", "accounting", "manager", "ceo"], "target_turns": (3, 6)
    },
    {
        "stage": 2, "enemy": "Cyber Sentinel", "target": "algorithm", "domain": "Technology", "hp": 200, 
        "guesses": ["computer", "code", "spreadsheet", "database", "source code"], "target_turns": (3, 6)
    },
    {
        "stage": 3, "enemy": "Alchemist Golem", "target": "catalyst", "domain": "Science", "hp": 200, 
        "guesses": ["physics", "chemistry", "molecule", "chemical reaction"], "target_turns": (2, 5)
    }
]

def load_domain_corpus(csv_path="corpus/domain_docs.csv"):
    """
    โหลดคอร์ปัสรูปแบบใหม่ที่เป็นตารางเอกสารรายโดเมน (ตามโครงสร้างของกลุ่ม)
    """
    try:
        corpus_df = pd.read_csv(csv_path)
        print(f"Loaded domain corpus from {csv_path} ({len(corpus_df)} documents)")
        return corpus_df
    except FileNotFoundError:
        print(f"[!] Warning: Not found '{csv_path}'. Continuing without corpus loading.")
        return None

def calculate_dynamic_damage_tuned(player_input, target_word, embeddings, max_hp=100, cutoffs=(0.2, 0.4, 0.6, 0.8)):
    sim = get_cosine_similarity(player_input, target_word, embeddings)
    t1, t2, t3, t4 = cutoffs
    
    if sim < t1:
        damage = 0
    elif sim < t2:
        damage = int(max_hp * 0.10) # ปรับฐานดาเมจขั้นต่ำให้แรงขึ้นเล็กน้อย (10%)
    elif sim < t3:
        damage = int(max_hp * 0.20) # 20%
    elif sim < t4:
        damage = int(max_hp * 0.35) # 35%
    else:
        damage = int(max_hp * 0.50) # 50% (Critical)
        
    return damage

def evaluate_parameters(gate_threshold, damage_cutoffs, embeddings):
    # --- ส่วนที่ 1: ประเมิน Gate Accuracy ---
    correct_gate = 0
    for target, guess, domain, expected in GATE_TEST_CASES:
        phrase_vec = get_phrase_vector(guess, embeddings)
        if phrase_vec is None:
            pred = "out_of_domain"
        else:
            # ใช้ gate_threshold ที่กำลังจูน
            gate_res = soft_domain_gate(guess, domain, embeddings, threshold=gate_threshold)
            pred = "in_domain" if gate_res["allowed"] else "out_of_domain"
        
        if pred == expected:
            correct_gate += 1
    gate_acc = correct_gate / len(GATE_TEST_CASES)

    # --- ส่วนที่ 2: ประเมิน Pacing (Boss Defeated & Turns) ---
    pacing_success_count = 0
    total_stages = len(SIMULATION_STAGES)
    
    for stage in SIMULATION_STAGES:
        hp = stage["hp"]
        turns = 0
        used = set()
        
        for guess in stage["guesses"]:
            if hp <= 0 or guess in used:
                continue
            used.add(guess)
            
            gate_res = soft_domain_gate(guess, stage["domain"], embeddings, threshold=gate_threshold)
            if gate_res["allowed"]:
                base_dmg = calculate_dynamic_damage_tuned(guess, stage["target"], embeddings, max_hp=stage["hp"], cutoffs=damage_cutoffs)
                actual_dmg = int(base_dmg * gate_res["multiplier"])
                hp -= actual_dmg
                turns += 1
            else:
                turns += 1
                
        defeated = (hp <= 0)
        min_t, max_t = stage["target_turns"]
        if defeated and (min_t <= turns <= max_t):
            pacing_success_count += 1
            
    pacing_score = pacing_success_count / total_stages

    # คะแนนรวม (ให้น้ำหนัก Gate 40% และ Pacing 60%)
    total_score = (gate_acc * 0.4) + (pacing_score * 0.6)
    return total_score, gate_acc, pacing_score


def run_grid_search(embeddings):
    print("=" * 60)
    print("STARTING AUTOMATED GRID SEARCH & CALIBRATION...")
    print("=" * 60)
    
    # กำหนดช่วงค่าพารามิเตอร์ที่ต้องการทดลองค้นหา (Grid)
    threshold_grid = [0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30, 0.40]
    
    # ขยายช่วงตัดดาเมจให้หลากหลายขึ้น
    cutoff_grid = [
        (0.10, 0.20, 0.40, 0.60),
        (0.15, 0.30, 0.50, 0.70),
        (0.20, 0.40, 0.60, 0.80),
        (0.05, 0.15, 0.30, 0.50)
    ]
    
    best_score = -1
    best_params = None
    all_experiments = []
    
    total_combinations = len(threshold_grid) * len(cutoff_grid)
    current = 0
    
    for th, cut in itertools.product(threshold_grid, cutoff_grid):
        current += 1
        score, g_acc, p_score = evaluate_parameters(th, cut, embeddings)
        print(f"[{current}/{total_combinations}] Testing -> Threshold: {th}, Cutoffs: {cut} | Score: {score:.2f} (Gate Acc: {g_acc*100:.0f}%, Pacing Score: {p_score*100:.0f}%)")

        all_experiments.append({
            "Gate_Threshold": th,
            "Cutoff_T1": cut[0],
            "Cutoff_T2": cut[1],
            "Cutoff_T3": cut[2],
            "Cutoff_T4": cut[3],
            "Gate_Accuracy_Pct": g_acc * 100,
            "Pacing_Score_Pct": p_score * 100,
            "Total_Fitness_Score": score
        })
        
        if score > best_score:
            best_score = score
            best_params = {"threshold": th, "cutoffs": cut}
            
    print("=" * 60)
    print("CALIBRATION COMPLETED! BEST PARAMETERS FOUND:")
    print("=" * 60)
    print(f" Best Threshold for Soft Domain Gate: {best_params['threshold']}")
    print(f" Best Damage Cutoffs (t1, t2, t3, t4): {best_params['cutoffs']}")
    print(f" Highest Fitness Score Achieved: {best_score:.2f}")
    print("=" * 60)

    df_log = pd.DataFrame(all_experiments)
    df_log.to_csv("grid_search_results.csv", index=False, encoding="utf-8")
    print("\n📁 Saved Grid Search log as 'grid_search_results.csv' successfully!")
    
    return best_params

if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    embeddings = load_glove_model(glove_file)
    print("-" * 60)
    
    # โหลดคอร์ปัสเอกสารรายโดเมนเพื่อเตรียมใช้งานร่วมกับการประเมิน
    corpus_df = load_domain_corpus("corpus/domain_docs.csv")
    print("-" * 60)
    
    # รัน Grid Search เพื่อหาค่าที่ดีที่สุด
    optimal_params = run_grid_search(embeddings)