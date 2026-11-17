from glove_loader import load_glove_model, get_cosine_similarity

def test_command_matching():
    """
    Test the command matching system for the TextGame
    Even if the player types different words, GloVe will help match them to the main game commands
    """
    # 1. โหลดโมเดล GloVe 100d ที่เราเตรียมไว้
    glove_file = "glove.6B.100d.txt"
    embeddings = load_glove_model(glove_file)
    print("-" * 50)
    
    # คำสั่งหลักที่มีอยู่ในเกม TextGame ของคุณ
    game_commands = ["attack", "run", "heal", "inventory", "search"]
    
    # ลองจำลองคำที่ผู้เล่นพิมพ์เข้ามา (อาจไม่ใช่คำหลักตรงๆ แต่มีความหมายใกล้เคียง)
    player_inputs = ["strike", "escape", "potion", "bag", "explore"]
    
    print("Starting Semantic Command Matching test in the game:\n")
    
    for player_input in player_inputs:
        print(f"Player typed: '{player_input}'")
        
        best_match = None
        highest_sim = -1.0
        
        # เช็กความเหมือนกับทุกคำสั่งในเกม
        for cmd in game_commands:
            sim = get_cosine_similarity(player_input, cmd, embeddings)
            print(f"   -> comparing with '{cmd}': {sim:.4f}")
            
            if sim > highest_sim:
                highest_sim = sim
                best_match = cmd
                
        print(f"[System decision]: command **'{best_match}'** (similarity: {highest_sim:.4f})")
        print("-" * 50)

if __name__ == "__main__":
    test_command_matching()