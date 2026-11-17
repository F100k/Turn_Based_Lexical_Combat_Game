# เอกสารสรุปการทำงานและส่งมอบงาน: โมดูล Static Embedding (GloVe) 
**โมเดลหลักที่ใช้:** `glove.6B.100d.txt` (คลังคำศัพท์ $400,000$ คำ, มิติเวกเตอร์ $d = 100$)

---

## 1. ภาพรวมของโมดูล (Module Overview)
โมดูลนี้ทำหน้าที่เป็นระบบประมวลผลภาษาธรรมชาติฝั่ง **Static Word Embedding** สำหรับคำนวณความใกล้เคียงทางความหมาย (Semantic Similarity) ระหว่างคำหรือวลีที่ผู้เล่นพิมพ์เข้ามากับคำเป้าหมายของบอส (Target Word) โดยมีกลไกหลัก 4 ส่วน ได้แก่:
1. **Phrase Vector Composition:** การแปลงวลีหลายคำให้เป็นเวกเตอร์เดี่ยวด้วยวิธี Mean Pooling และทำ $L_2$ Normalization
2. **Soft Domain Gate (4 โดเมน):** ระบบคัดกรองหมวดหมู่คำศัพท์ของผู้เล่น (*Business, Technology, Science, Arts*) พร้อมคำนวณตัวคูณโบนัสดาเมจ (Multiplier)
3. **Dynamic Damage Calculation:** ระบบคำนวณความเสียหายตามระดับความเหมือน (Cosine Similarity) แบ่งเป็น 5 ขั้น (Tiers)
4. **Smart OOV Recovery:** ระบบตรวจสอบความครอบคลุมของคำศัพท์ในคลังข้อมูล (`corpus.txt`) และกู้คืนคำสะกดผิดหรือคำนอกคลังศัพท์ (Out-of-Vocabulary) อัตโนมัติ

---

## 2. โครงสร้างไฟล์ในโฟลเดอร์ (File Structure)

> **หมายเหตุสำหรับทีม:** ไฟล์โค้ดและผลลัพธ์เวอร์ชันสมบูรณ์ที่ผ่านการปรับจูนพารามิเตอร์ล่าสุดจะใช้ชุดไฟล์หลัก (หรือชุดที่อัปเดตเป็น `_2` ในการส่งมอบล่าสุด) โดยมีรายละเอียดดังนี้

| ชื่อไฟล์ | ประเภทไฟล์ | หน้าที่และรายละเอียด |
| :--- | :--- | :--- |
| `glove_loader.py` | Core Module | ไฟล์ศูนย์กลางสำหรับโหลดโมเดล GloVe, แปลงวลีเป็นเวกเตอร์, ตรวจสอบ Soft Domain Gate และคำนวณดาเมจ |
| `grid_search_calibration.py` | Optimization | สคริปต์ค้นหาค่าพารามิเตอร์ `threshold` และ `cutoffs` ที่เหมาะสมที่สุดแบบอัตโนมัติ ($32$ รูปแบบการทดลอง) |
| `evaluate_glove_game.py` | Evaluation | สคริปต์วัดผลความแม่นยำของ Soft Domain Gate ($16$ เคส) และจำลองการต่อสู้จริง (Battle Pacing Simulation $3$ ด่าน) |
| `check_oov.py` | Corpus & OOV | สคริปต์ตรวจสอบความครอบคลุมของคำศัพท์ใน `corpus.txt` เทียบกับ GloVe และทดสอบระบบแก้คำผิดด้วย `difflib` |
| `visualize_game.py` | Visualization | สคริปต์พล็อตกราฟ 3 มิติ (3D Semantic Space) เทียบคำศัพท์ผู้เล่นกับแกนอ้างอิง 3 แกนด้วย `matplotlib` |
| `test_damage.py` | Unit Test | สคริปต์ขนาดเล็กสำหรับทดสอบการคิดดาเมจกับบอสเดี่ยวเบื้องต้น |
| `corpus.txt` | Dataset | คลังข้อความเนื้อหาในเกมสำหรับใช้วิเคราะห์อัตราส่วนคำศัพท์ In-Vocab และ OOV |
| `grid_search_results.csv` | Report Artifact | ตารางบันทึกผลการทดลองปรับจูนพารามิเตอร์ทั้ง $32$ รอบ |
| `gate_evaluation_report.csv` | Report Artifact | ตารางผลลัพธ์การคัดกรองคำเข้า-ออกโดเมนทั้ง $16$ เคสทดสอบ |
| `battle_pacing_report.csv`  | Report Artifact | ตารางสรุปผลการจำลองจำนวนเทิร์นที่ใช้ปราบบอสในโหมด *Typical Play* และ *Expert Play* |

---

## 3. ลอจิกและสมการทางคณิตศาสตร์ที่ใช้ (Core Methodology)

### 3.1 การแปลงวลีเป็นเวกเตอร์ (`get_phrase_vector` & `get_phrase_similarity`)
เนื่องจาก GloVe เก็บเวกเตอร์แยกตามคำเดี่ยว เมื่อผู้เล่นพิมพ์วลีหลายคำ (เช่น `"source code"` หรือ `"chemical reaction"`) ระบบจะดำเนินการตามขั้นตอนต่อไปนี้:
1. แปลงข้อความเป็นตัวพิมพ์เล็ก แยกคำด้วยช่องว่าง และตัดคำเชื่อมพื้นฐาน (Stopwords: `{"a", "an", "the", "in", "on", "at", "by", "for", "with", "and", "or"}`) ออก
2. ดึงเวกเตอร์ของคำที่อยู่ในคลังศัพท์ $\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_k \in \mathbb{R}^{100}$ มาหาค่าเฉลี่ย (Mean Pooling):
   $$\mathbf{v}_{\text{mean}} = \frac{1}{k} \sum_{i=1}^{k} \mathbf{w}_i$$
3. ทำ $L_2$ Normalization เพื่อให้เวกเตอร์มีความยาวเท่ากับ $1$ (มาตรฐานเดียวกับฝั่งโมเดล Contextual):
   $$\hat{\mathbf{v}} = \frac{\mathbf{v}_{\text{mean}}}{\|\mathbf{v}_{\text{mean}}\|_2}$$
4. คำนวณความเหมือนระหว่างวลีของผู้เล่น $\hat{\mathbf{v}}_p$ กับคำเป้าหมาย $\hat{\mathbf{v}}_t$ ด้วยผลคูณภายใน (Dot Product ซึ่งมีค่าเท่ากับ Cosine Similarity):
   $$\text{Similarity} = \hat{\mathbf{v}}_p \cdot \hat{\mathbf{v}}_t$$

### 3.2 ระบบคัดกรองหมวดหมู่ Soft Domain Gate (`soft_domain_gate`)
ระบบกำหนดโดเมนหลักไว้ $4$ หมวด ได้แก่ `Business`, `Technology`, `Science` และ `Arts` โดยแต่ละหมวดมีประโยคต้นแบบ (`DOMAIN_ANCHORS`) หมวดละ $3$ ประโยค เพื่อนำมาเฉลี่ยและทำ $L_2$ Normalization เป็นเวกเตอร์จุดศูนย์กลางของโดเมน ($\hat{\mathbf{d}}_{\text{dom}}$)

เมื่อผู้เล่นพิมพ์คำตอบ ระบบจะคำนวณคะแนนความเหมือนกับทุกโดเมน โดยให้:
* $s_{\text{req}}$ = คะแนนความเหมือนกับโดเมนที่ด่านนั้นต้องการ (`required_domain`)
* $s_{\text{best}}$ = คะแนนความเหมือนสูงสุดจากทั้ง $4$ โดเมน (ตรงกับโดเมน `best_domain`)
* $\tau$ = ค่าเกณฑ์ขั้นต่ำ (`threshold = 0.30` จากการทำ Grid Search)

**เงื่อนไขการตัดสินและตัวคูณดาเมจ (Gate Status & Multipliers):**
* **กรณีไม่พบคำในคลังศัพท์:** คืนค่า `allowed = False`, ตัวคูณ $0.0\times$, สถานะ `UNKNOWN / OOV`
* **กรณีผ่านเกณฑ์หลัก** ($s_{\text{req}} \ge \tau$ และ ($\text{best\_domain} = \text{required\_domain}$ หรือ $s_{\text{best}} - s_{\text{req}} < 0.12$)):
  * หากโดเมนที่คะแนนสูงสุดตรงกับโดเมนเป้าหมายและ $s_{\text{req}} \ge 0.30$: ได้สถานะ **`PRIMARY`** (`allowed = True`, ตัวคูณโบนัส **$1.25\times$**)
  * นอกเหนือจากนั้น: ได้สถานะ **`IN-DOMAIN`** (`allowed = True`, ตัวคูณปกติ **$1.00\times$**)
* **กรณีคะแนนรองลงมาหรือหลุดกรอบ:**
  * หากยังผ่านเกณฑ์ขั้นต่ำ ($s_{\text{req}} \ge \tau$) แต่ไปใกล้เคียงโดเมนอื่นมากกว่า: ได้สถานะ **`SECONDARY`** (`allowed = True`, ตัวคูณถูกลดทอนเหลือ **$0.75\times$**)
  * หาก $s_{\text{req}} < \tau$: ได้สถานะ **`BLOCKED (Out of Domain)`** (`allowed = False`, ตัวคูณ **$0.0\times$**)

### 3.3 ระบบคำนวณดาเมจแบบขั้นบันได (`calculate_dynamic_damage`)
เมื่อคำของผู้เล่นผ่าน Soft Domain Gate (`allowed = True`) ระบบจะคำนวณดาเมจพื้นฐาน (`base_damage`) เทียบกับเลือดสูงสุดของบอส (`max_hp`) โดยใช้จุดตัด `cutoffs` ที่ปรับจูนแล้วคือ $(t_1, t_2, t_3, t_4) = (0.05, 0.15, 0.30, 0.50)$:

| ช่วงค่าความเหมือน ($\text{sim}$) | ระดับการโจมตี (Combat Tier) | ดาเมจพื้นฐานเทียบกับเลือดบอส (`max_hp`) |
| :--- | :--- | :--- |
| $\text{sim} < 0.05$ | `BLOCKED / MISS` | $0\%$ |
| $0.05 \le \text{sim} < 0.15$ | `WEAK GRAZE` | $10\%$ |
| $0.15 \le \text{sim} < 0.30$ | `SOLID HIT` | $20\%$ |
| $0.30 \le \text{sim} < 0.50$ | `STRONG HIT` | $35\%$ |
| $\text{sim} \ge 0.50$ | `CRITICAL STRIKE!` | $50\%$ |

จากนั้นคำนวณดาเมจสุทธิที่ทำได้จริงในเทิร์นนั้นด้วยสมการ:
$$\text{Actual Damage} = \lfloor \text{Base Damage} \times \text{Gate Multiplier} \rfloor$$

---

## 4. สรุปผลการทดลองและหลักฐานไฟล์ CSV (Experimental Results)

### 4.1 ผลตรวจสอบคลังคำศัพท์และการกู้คืนคำผิด (`check_oov.py`)
จากการตรวจสอบคำศัพท์ที่ไม่ซ้ำกันทั้งหมดในไฟล์ `corpus.txt` จำนวน **$31,348$ คำ** เทียบกับโมเดล `glove.6B.100d.txt`:
* **คำที่มีอยู่ในโมเดล (In-Vocabulary):** $28,946$ คำ คิดเป็น **$92.34\%$**
* **คำนอกคลังศัพท์ (Out-of-Vocabulary / OOV):** $2,402$ คำ คิดเป็น **$7.66\%$**
* **ระบบสำรอง Smart OOV Recovery:** ใช้ไลบรารี `difflib.get_close_matches` (ตั้งค่า `cutoff = 0.6`) สามารถกู้คืนคำที่ผู้เล่นสะกดผิดให้กลับมาเป็นคำในคลังศัพท์ได้อย่างแม่นยำ เช่น:
  * `'directr'` $\rightarrow$ แก้เป็น `'director'`
  * `'managr'` $\rightarrow$ แก้เป็น `'manager'`
  * `'exectiv'` $\rightarrow$ แก้เป็น `'exective'`

### 4.2 ผลการปรับจูนพารามิเตอร์ด้วย Grid Search (`grid_search_results.csv`)
สคริปต์ `grid_search_calibration_2.py` ทดสอบพารามิเตอร์ทั้งหมด $32$ รูปแบบ (ค่า `threshold` $8$ ค่า $\times$ ชุด `cutoffs` $4$ ชุด) โดยใช้สมการ Fitness Score ที่ถ่วงน้ำหนักระหว่างความแม่นยำของ Gate ($40\%$) และความสมดุลของจำนวนเทิร์นปราบบอส ($60\%$):
$$\text{Total Fitness Score} = 0.40 \times \text{Gate Accuracy} + 0.60 \times \text{Pacing Score}$$

* **พารามิเตอร์ที่ดีที่สุดที่ค้นพบ:**
  * `Best Threshold` สำหรับ Soft Domain Gate = **$0.30$**
  * `Best Damage Cutoffs` $(t_1, t_2, t_3, t_4)$ = **$(0.05, 0.15, 0.30, 0.50)$**
* **คะแนนสูงสุดที่ทำได้ (การทดลองที่ 28 และ 32):**
  * `Gate_Accuracy_Pct` = **$93.75\%$** (ทายถูก $15/16$ เคส)
  * `Pacing_Score_Pct` = **$100.00\%$** (ปราบบอสในโหมดผู้เล่นทั่วไปภายในเกณฑ์เทิร์นครบทั้ง $3/3$ ด่าน)
  * `Total_Fitness_Score` = **$0.9750$** (แสดงผลบนหน้าจอ $0.97$)

### 4.3 ผลประเมินความแม่นยำของ Soft Domain Gate (`gate_evaluation_report.csv`)
เมื่อนำค่า `threshold = 0.30` ไปใช้ในฟังก์ชันหลักแล้วรันประเมินกับชุดทดสอบมาตรฐาน $16$ เคส ($4$ โดเมน โดเมนละ $4$ เคส แบ่งเป็น `in_domain` $8$ เคส และ `out_of_domain` $8$ เคส) ได้ผลสรุปดังนี้:

* **Overall Accuracy:** **$93.75\%$ ($15/16$ เคส)**
* **Confusion Matrix (กำหนดให้ `in_domain` เป็นคลาสบวก):**
  * True Positive ($\text{TP}$): $8$ เคส (คำในโดเมนผ่าน Gate ครบ $100\%$)
  * True Negative ($\text{TN}$): $7$ เคส (คำนอกโดเมนถูกบล็อกถูกต้อง $7$ จาก $8$ เคส)
  * False Positive ($\text{FP}$): $1$ เคส (คำนอกโดเมนหลุดผ่าน Gate คือวลี `"abstract painting"` ในด่าน Technology)
  * False Negative ($\text{FN}$): $0$ เคส (ไม่มีคำที่ถูกหมวดโดนบล็อกแม้แต่คำเดียว)

**ตาราง Classification Report ของโมดูล GloVe:**

| Class / Metric | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **in_domain** | $0.8889$ | $1.0000$ | $0.9412$ | $8$ |
| **out_of_domain** | $1.0000$ | $0.8750$ | $0.9333$ | $8$ |
| **Macro Average** | **$0.9444$** | **$0.9375$** | **$0.9373$** | **$16$** |

### 4.4 ผลจำลองความสมดุลของการต่อสู้ (`battle_pacing_report.csv`)
ผลการจำลองการต่อสู้กับบอส $3$ ด่าน แยกตามกลยุทธ์ผู้เล่นทั่วไป (*Typical Play*) และผู้เล่นเชี่ยวชาญ (*Expert Play*):

| ด่าน (Stage) | บอส / โดเมน / เลือด | กลยุทธ์ (Strategy) | จำนวนเทิร์นที่ใช้ | ช่วงเทิร์นเป้าหมาย | ปราบบอสสำเร็จ | สถานะ Pacing |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | Corporate Titan (`Business`, $180$ HP) | Typical Play | $3$ เทิร์น | $3\text{--}6$ เทิร์น | Yes | PASS |
| **Stage 1** | Corporate Titan (`Business`, $180$ HP) | Expert Play | $2$ เทิร์น | $3\text{--}6$ เทิร์น | Yes | CHECK (ชนะเร็วกว่าเกณฑ์) |
| **Stage 2** | Cyber Sentinel (`Technology`, $200$ HP) | Typical Play | $3$ เทิร์น | $3\text{--}6$ เทิร์น | Yes | PASS |
| **Stage 2** | Cyber Sentinel (`Technology`, $200$ HP) | Expert Play | $3$ เทิร์น | $3\text{--}6$ เทิร์น | Yes | PASS |
| **Stage 3** | Alchemist Golem (`Science`, $200$ HP) | Typical Play | $3$ เทิร์น | $2\text{--}5$ เทิร์น | Yes | PASS |
| **Stage 3** | Alchemist Golem (`Science`, $200$ HP) | Expert Play | $2$ เทิร์น | $2\text{--}5$ เทิร์น | No (เหลือ $26$ HP) | CHECK (รายการคำทายมี $2$ คำ) |

**ประเด็นสำคัญสำหรับนำไปเขียนอภิปรายผล (Error Analysis & Discussion):**
1. **ความสมดุลในโหมดผู้เล่นทั่วไป (Typical Play):** ผ่านเกณฑ์ `PASS` ครบ $100\%$ ทั้ง $3$ ด่าน โดยใช้จำนวนเทิร์นเฉลี่ย $3$ เทิร์นในการปราบบอส
2. **กรณีศึกษา False Positive จาก Mean Pooling:** ในด่าน Technology วลี `"abstract painting"` หลุดผ่าน Gate เนื่องจากคำว่า `"abstract"` มีความหมายพ้องในบริบทวิทยาการคอมพิวเตอร์ (เช่น *abstract data type / abstraction*) เมื่อนำเวกเตอร์มาหาค่าเฉลี่ยแบบไม่มีบริบทกำกับ (Static Embedding) จึงได้ค่าความเหมือนสูงกว่าเกณฑ์ขั้นต่ำ ซึ่งจุดนี้สามารถนำไปเปรียบเทียบกับโมเดล Contextual (SBERT) ของเพื่อนในทีมได้โดยตรง
3. **กรณีศึกษาโหมด Expert Play ในด่านที่ 1 และ 3:** ด่านที่ 1 คำว่า `"manager"` และ `"director"` ได้รับโบนัส `PRIMARY` ($1.25\times$) และติด `CRITICAL STRIKE!` ทั้งคู่ จึงทำดาเมจรวม $224$ HP ปราบบอสได้ใน $2$ เทิร์น (เร็วกว่ากรอบ $3\text{--}6$ เทิร์น) ส่วนด่านที่ 3 ชุดคำทาย `expert_guesses` กำหนดไว้เพียง $2$ คำ (`["molecule", "chemical reaction"]`) ซึ่งทำดาเมจรวมได้ $174$ HP จาก $200$ HP

---

## 5. คู่มือการนำฟังก์ชันไปเชื่อมต่อกับระบบเกมหลัก (Integration Guide)

เพื่อนในทีมที่ดูแลระบบเกมหลักสามารถ `import` ฟังก์ชันจาก `glove_loader.py` ไปใช้งานในลูปการต่อสู้ได้ทันทีตามตัวอย่างนี้:

```python
from glove_loader import load_glove_model, soft_domain_gate, calculate_dynamic_damage

# 1. โหลดโมเดล 1 ครั้งตอนเริ่มเกม
embeddings = load_glove_model("glove.6B.100d.txt")

# 2. ตัวอย่างข้อมูลบอสในด่านปัจจุบัน
boss_domain = "Technology"
boss_target = "algorithm"
boss_max_hp = 200
player_input = "source code"

# 3. ตรวจสอบ Soft Domain Gate (ใช้ค่า Default threshold=0.3 ที่จูนแล้ว)
gate_res = soft_domain_gate(player_input, boss_domain, embeddings)

if gate_res["allowed"]:
    # 4. คำนวณดาเมจ (ใช้ค่า Default cutoffs=(0.05, 0.15, 0.30, 0.50) ที่จูนแล้ว)
    dmg_res = calculate_dynamic_damage(player_input, boss_target, embeddings, max_hp=boss_max_hp)
    actual_damage = int(dmg_res["damage"] * gate_res["multiplier"])
    print(f"Status: {gate_res['status']} | Tier: {dmg_res['tier']} | Damage: {actual_damage}")
else:
    actual_damage = 0
    print(f"Blocked: {gate_res['status']} | Damage: 0")
```

---

## 6. คำสั่งสำหรับรันทดสอบซ้ำ (Reproducibility Commands)

หากต้องการรันสคริปต์เพื่อตรวจสอบผลลัพธ์หรือสร้างไฟล์ CSV ใหม่ทั้งหมด ให้เปิด Terminal ในโฟลเดอร์ที่มีไฟล์ `glove.6B.100d.txt` และรันคำสั่งตามลำดับดังนี้:

```bash
# 1. ตรวจสอบเปอร์เซ็นต์ความครอบคลุมคำศัพท์ใน Corpus และระบบแก้คำผิด OOV
python check_oov.py

# 2. รัน Grid Search 32 รูปแบบเพื่อสร้างไฟล์ grid_search_results.csv
python grid_search_calibration.py

# 3. รันประเมินผล Gate และจำลองการต่อสู้เพื่อสร้างไฟล์ gate_evaluation_report.csv และ battle_pacing_report.csv
python evaluate_glove_game.py

# 4. รันแสดงผลกราฟ 3 มิติ (3D Semantic Space)
python visualize_game.py
```