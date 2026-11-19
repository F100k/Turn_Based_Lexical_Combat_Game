import numpy as np
import pandas as pd
from glove_loader import load_glove_model, get_phrase_vector, soft_domain_gate, calculate_dynamic_damage, load_domain_corpus

# ปรับปรุงเคสทดสอบให้ตรงกับ 4 โดเมนหลักของเรา (Business, Technology, Science, History)
GATE_TEST_CASES = [
    # (Target, Player Guess, Required Domain, Expected Gate Label)
    ("executive", "ceo", "Business", "in_domain"),
    ("executive", "director", "Business", "in_domain"),
    ("executive", "banana", "Business", "out_of_domain"),
    ("executive", "algorithm", "Business", "out_of_domain"),
    
    ("algorithm", "source code", "Technology", "in_domain"),
    ("algorithm", "database", "Technology", "in_domain"),
    ("algorithm", "chemical reaction", "Technology", "out_of_domain"),
    ("algorithm", "ceo", "Technology", "out_of_domain"),
    
    ("catalyst", "molecule", "Science", "in_domain"),
    ("catalyst", "chemical reaction", "Science", "in_domain"),
    ("catalyst", "spreadsheet", "Science", "out_of_domain"),
    ("catalyst", "history", "Science", "out_of_domain"),
    
    # ปรับจาก Arts เป็น History หรือหมวดหลักในระบบ
    ("revolution", "war", "History", "in_domain"),
    ("revolution", "empire", "History", "in_domain"),
    ("revolution", "algorithm", "History", "out_of_domain"),
    ("revolution", "banana", "History", "out_of_domain"),
]

SIMULATION_STAGES = [
    {
        "stage": 1, 
        "enemy": "Corporate Titan", 
        "target": "executive", 
        "domain": "Business", 
        "hp": 180, 
        "expert_guesses": ["manager", "director", "ceo"], 
        "typical_guesses": ["banana", "employee", "accounting", "manager", "ceo"], 
        "target_turns": (3, 6) 
    },
    {
        "stage": 2, 
        "enemy": "Cyber Sentinel", 
        "target": "algorithm", 
        "domain": "Technology", 
        "hp": 200, 
        "expert_guesses": ["software", "database", "source code"], 
        "typical_guesses": ["computer", "code", "spreadsheet", "database", "source code"], 
        "target_turns": (3, 6)
    },
    {
        "stage": 3, 
        "enemy": "Alchemist Golem", 
        "target": "catalyst", 
        "domain": "Science", 
        "hp": 200, 
        "expert_guesses": ["molecule", "chemical reaction"], 
        "typical_guesses": ["physics", "chemistry", "molecule", "chemical reaction"], 
        "target_turns": (2, 5)
    }
]

def evaluate_gate_performance(embeddings):
    """
    ประเมินประสิทธิภาพของ Soft Domain Gate ว่าทายถูกกี่เปอร์เซ็นต์
    """
    print("=" * 60)
    print("1. EVALUATING SOFT DOMAIN GATE PERFORMANCE")
    print("=" * 60)
    
    correct_predictions = 0
    results = []
    
    for target, guess, domain, expected_label in GATE_TEST_CASES:
        gate_res = soft_domain_gate(guess, domain, embeddings)
        
        if gate_res["allowed"]:
            predicted_label = "in_domain"
        else:
            predicted_label = "out_of_domain"
            
        is_correct = (predicted_label == expected_label)
        if is_correct:
            correct_predictions += 1
            
        results.append({
            "Domain": domain,
            "Target": target,
            "Guess": guess,
            "Expected": expected_label,
            "Predicted": predicted_label,
            "Status": "✅ PASS" if is_correct else "❌ FAIL"
        })
        
    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))

    df_results.to_csv("gate_evaluation_report.csv", index=False, encoding="utf-8")
    print("📁 Saved: 'gate_evaluation_report.csv'")
    
    accuracy = (correct_predictions / len(GATE_TEST_CASES)) * 100
    print("-" * 60)
    print(f"Gate Classification Accuracy: {accuracy:.2f}% ({correct_predictions}/{len(GATE_TEST_CASES)})")
    print("=" * 60 + "\n")
    return accuracy

def simulate_battle_pacing(embeddings):
    """
    จำลองการต่อสู้ (Battle Simulation) เพื่อเช็ก Pacing ว่าผู้เล่นใช้กี่ตาถึงชนะบอส
    """
    print("=" * 60)
    print("2. SIMULATING BATTLE PACING & DAMAGE SCALING")
    print("=" * 60)
    
    simulation_summaries = []
    
    for stage in SIMULATION_STAGES:
        print(f"\n--- Stage {stage['stage']}: {stage['enemy']} (HP: {stage['hp']}, Domain: {stage['domain']}) ---")
        
        for play_type, guesses in [("Typical Play", stage["typical_guesses"]), ("Expert Play", stage["expert_guesses"])]:
            hp = stage["hp"]
            turns_used = 0
            used_words = set()
            
            for guess in guesses:
                if hp <= 0:
                    break
                if guess in used_words:
                    continue
                used_words.add(guess)
                
                gate_res = soft_domain_gate(guess, stage["domain"], embeddings)
                
                if gate_res["allowed"]:
                    dmg_res = calculate_dynamic_damage(guess, stage["target"], embeddings, max_hp=stage["hp"])
                    actual_dmg = int(dmg_res["damage"] * gate_res["multiplier"])
                    hp -= actual_dmg
                    turns_used += 1
                    print(f"  [{play_type}] Guess '{guess}' -> Gate: {gate_res['status']} | Tier: {dmg_res['tier']} | Dealt: -{actual_dmg} HP (Boss HP left: {max(0, hp)})")
                else:
                    turns_used += 1
                    print(f"  [{play_type}] Guess '{guess}' -> Gate: BLOCKED | Dealt: -0 HP")
            
            defeated = (hp <= 0)
            min_t, max_t = stage["target_turns"]
            pacing_met = (min_t <= turns_used <= max_t) if defeated else False
            
            simulation_summaries.append({
                "Stage": stage["stage"],
                "Strategy": play_type,
                "Turns Taken": turns_used,
                "Target Range": f"{min_t}-{max_t} turns",
                "Boss Defeated": "✅ Yes" if defeated else "❌ No",
                "Pacing Balanced": "✅ PASS" if pacing_met else "⚠️ CHECK"
            })
            
    print("\n" + "-" * 60)
    print("BATTLE PACING SUMMARY REPORT:")
    print("-" * 60)
    df_pacing = pd.DataFrame(simulation_summaries)
    print(df_pacing.to_string(index=False))
    
    df_pacing.to_csv("battle_pacing_report.csv", index=False, encoding="utf-8")
    print("📁 Saved: 'battle_pacing_report.csv'")
    print("=" * 60)

if __name__ == "__main__":
    glove_file = "glove.6B.100d.txt"
    
    print("Initializing GloVe model for Simulation & Evaluation...")
    embeddings = load_glove_model(glove_file)
    print("-" * 60)
    
    # โหลดคอร์ปัสเอกสารรายโดเมน CSV เพื่อความสมบูรณ์
    corpus_df = load_domain_corpus("corpus/domain_docs.csv")
    print("-" * 60)
    
    evaluate_gate_performance(embeddings)
    simulate_battle_pacing(embeddings)