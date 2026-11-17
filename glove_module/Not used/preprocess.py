import re
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def load_and_preprocess_corpus(file_path="corpus.txt", min_word_freq=2):
    
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    tokenized_corpus = []
    word_counts = Counter()
    
    # 1. อ่านไฟล์และทำการ Tokenization
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            # ดึงเฉพาะคำภาษาอังกฤษตัวพิมพ์เล็ก
            tokens = re.findall(r'\b[a-z]+\b', line.lower())
            
            # กรอง Stopwords และทำ Lemmatization
            processed_tokens = [
                lemmatizer.lemmatize(word) 
                for word in tokens 
                if word not in stop_words and len(word) > 2
            ]
            
            if processed_tokens:
                tokenized_corpus.append(processed_tokens)
                word_counts.update(processed_tokens)
                
    # 2. กรองคำความถี่ต่ำและสร้าง Vocabulary
    filtered_corpus = [
        [word for word in sentence if word_counts[word] >= min_word_freq]
        for sentence in tokenized_corpus
    ]
    
    vocab = {word for sentence in filtered_corpus for word in sentence}
    
    print(f"Preprocessing completed successfully! (Sentences: {len(filtered_corpus):,}, Vocabulary: {len(vocab):,})\n")
    return filtered_corpus, vocab

if __name__ == "__main__":
    corpus, vocab = load_and_preprocess_corpus()
    print("Example Vocab:", list(vocab)[:15])