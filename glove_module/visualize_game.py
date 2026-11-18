import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from glove_loader import load_glove_model, get_cosine_similarity

def visualize_multiple_words(word_list, reference_axes, embeddings):
    """
    Function to compare the player's input word with the 'semantic 3 axes' in the game
    and plot it as a 3D scatter plot
    """
    print(f"Analyzing the dimensions of all {len(word_list)} words...")
    
    valid_words = []
    xs, ys, zs, colors = [], [], [], []
    
    for word in word_list:
        if word not in embeddings:
            print(f"  [!] Skipping word '{word}' (not in GloVe)")
            continue
            
        valid_words.append(word)
        
        # คำนวณ Cosine Similarity เทียบกับ 3 แกนหลัก (แปลงเป็นเปอร์เซ็นต์ 0-100%)
        scores = []
        for axis_word in reference_axes:
            sim = get_cosine_similarity(word, axis_word, embeddings)
            pct = max(0.0, sim) * 100
            scores.append(pct)
            
        xs.append(scores[0])
        ys.append(scores[1])
        zs.append(scores[2])
        
        # ใช้คะแนนของแกน Z (ความลับ) เป็นตัวกำหนดเฉดสี (Color intensity)
        colors.append(scores[2])

    if not valid_words:
        print("No valid words to analyze")
        return

    # สร้างกราฟ 3 มิติ
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # พล็อตจุดทั้งหมดแบบ 3D Scatter พร้อม Color Map (cmap)
    sc = ax.scatter(xs, ys, zs, c=colors, cmap='viridis', s=130, edgecolor='k', alpha=0.9)

    # โยงเส้นดิ่งลงพื้น (Stem lines) ให้เหมือนกราฟสถิติในรูปตัวอย่าง
    for x, y, z, word in zip(xs, ys, zs, valid_words):
        ax.plot([x, x], [y, y], [0, z], color='gray', linestyle='--', alpha=0.4, linewidth=1)
        # ใส่ชื่อคำศัพท์กำกับข้างๆ จุด
        ax.text(x + 1, y + 1, z + 1, f" {word}", fontsize=9, fontweight='bold', color='darkblue')

    # ตั้งชื่อแกนทั้ง 3 มิติ (ธีม Cyber/Mystery)
    ax.set_xlabel(f'Axis X: {reference_axes[0]}', fontweight='bold', labelpad=10)
    ax.set_ylabel(f'Axis Y: {reference_axes[1]}', fontweight='bold', labelpad=10)
    ax.set_zlabel(f'Axis Z: {reference_axes[2]}', fontweight='bold', labelpad=10)
    
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_zlim(0, 100)

    # เพิ่มแถบสี (Colorbar) ด้านข้างแบบในภาพตัวอย่าง
    cbar = plt.colorbar(sc, pad=0.1)
    cbar.set_label('Secret Score Intensity (Z-Axis Level)', fontweight='bold')

    plt.title("3D Semantic Dimension Space (Cyber/Mystery Theme)", fontsize=14, fontweight='bold', pad=20)
    plt.show()

if __name__ == "__main__":
    # 1. โหลด GloVe
    glove_file = "glove.6B.100d.txt"
    embeddings = load_glove_model(glove_file)
    
    # 2. เปลี่ยนแกนเรื่องเป็นแนวไซไฟสืบสวน (Cyber Mystery)
    cyber_mystery_axes = ["technology", "danger", "mystery"]
    
    # 3. กำหนดรายการคำศัพท์หลายๆ แบบที่จะเอามาเปรียบเทียบพร้อมกัน
    test_words = [
        "hacker", "gun", "shadow", "cipher", "android", 
        "poison", "neon", "data", "agent", "virus", 
        "ghost", "lock", "terminal", "weapon", "secret"
    ]
    
    # 4. รันฟังก์ชันแสดงผลกราฟ 3 มิติ
    visualize_multiple_words(test_words, cyber_mystery_axes, embeddings)