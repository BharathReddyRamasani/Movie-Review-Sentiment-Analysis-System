# 🎬 Movie Review Sentiment Analysis System

> **Deep Learning Based Sentiment Classification** using SimpleRNN, LSTM, and GRU models trained on the IMDB dataset.

---

## 📌 Overview

This project implements and compares three recurrent neural network architectures for sentiment analysis on movie reviews:

| Model | Architecture | Test Accuracy | Notes |
|-------|-------------|---------------|-------|
| SimpleRNN | Basic RNN | ~54% | Struggles with long-range dependencies |
| LSTM | Long Short-Term Memory | ~84.5% | Good at capturing context |
| **GRU** ⭐ | Gated Recurrent Unit | **~86.6%** | **Best performer** |

---

## 🚀 Features

- **Single Model Analysis** — Analyze a review with a specific model
- **Model Comparison Dashboard** — Compare all three models side-by-side
- **Confidence Scoring** — Visual probability bars and gauges
- **Consensus Voting** — See which sentiment majority models agree on
- **Sample Reviews** — Load positive/negative samples to test instantly

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8+
- pip

### 1. Clone the repository
```bash
git clone https://github.com/BharathReddyRamasani/Movie-Review-Sentiment-Analysis-System.git
cd Movie-Review-Sentiment-Analysis-System
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your trained model files
Place the following files in the root directory:
```
simplernn_imdb_model.keras
lstm_imdb_model.keras
gru_imdb_model.keras
imdb_word_index.pkl
imdb_maxlen.pkl
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```

---

## 📂 Project Structure

```
Movie-Review-Sentiment-Analysis-System/
│
├── app.py                          # 🎯 Main Streamlit application
├── LSTM_&__RNN_&_GRU.ipynb         # 📓 Model training notebook
├── requirements.txt                # 📦 Python dependencies
├── README.md                       # 📖 This file
│
├── simplernn_imdb_model.keras      # 🤖 SimpleRNN model (add after training)
├── lstm_imdb_model.keras           # 🔄 LSTM model (add after training)
├── gru_imdb_model.keras            # ⚡ GRU model (add after training)
├── imdb_word_index.pkl             # 📚 Word index mapping
└── imdb_maxlen.pkl                 # 📏 Sequence max length
```

---

## 🧠 Model Architecture

### SimpleRNN
```
Embedding(vocab_size, 32) → SimpleRNN(32) → Dense(1, sigmoid)
```

### LSTM
```
Embedding(vocab_size, 64) → LSTM(64) → Dense(1, sigmoid)
```

### GRU ⭐ (Best Performer)
```
Embedding(vocab_size, 32) → GRU(32) → Dense(1, sigmoid)
```

---

## 🎯 Usage

1. **Open** the Streamlit app in your browser
2. **Select Mode**: Single Model or Compare All Models
3. **Select Model** (if Single Model mode): SimpleRNN, LSTM, or GRU
4. **Enter** your movie review in the text area
5. **Click** "Analyze Review"
6. **View** sentiment prediction, confidence score, and probability distribution

---

## 📊 Dataset

- **IMDB Movie Reviews Dataset**: 50,000 reviews (25k train, 25k test)
- **Binary Classification**: Positive (1) or Negative (0)
- **Vocabulary Size**: 10,000 most frequent words
- **Sequence Length**: Padded/truncated to max length

---

## 🏆 Results

The **GRU model** achieved the best performance with **~86.6% accuracy** on the test set, demonstrating the effectiveness of gated recurrent architectures for sentiment classification tasks.

---

## 📦 Dependencies

- `streamlit` — Web app framework
- `tensorflow` — Deep learning framework
- `numpy` — Numerical computing

---

## 👨‍💻 Author

**Bharath Reddy Ramasani**

---

## 📝 License

MIT License