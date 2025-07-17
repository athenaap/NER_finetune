# Azerbaijani Named Entity Recognition (NER) Model

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

A pre-trained model for recognizing named entities in Azerbaijani text, supporting 25+ entity types including persons, locations, organizations, dates, and more.

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/azerbaijani-ner.git
cd azerbaijani-ner

# Install dependencies
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🚀 Quick Start

### Using the Command Line

```bash
# Analyze single text
python main.py --text "Apple Inc. Bakıda yeni ofis açacaq"

# Analyze a text file
python main.py --file input.txt

# Interactive mode
python main.py
```

### As a Python Module

```python
from ner_predictor import NERPredictor

model = NERPredictor()  # Automatically downloads model
results = model.predict("İlham Əliyev 2025-ci ildə Gəncədə görüş keçirəcək")
```

## 📊 Supported Entity Types

| Entity Type       | Example                |
|-------------------|------------------------|
| PERSON           | İlham Əliyev          |
| LOCATION         | Bakı, Xəzər dənizi    |
| ORGANIZATION     | Apple Inc.            |
| DATE             | 2024-cü ilin martı    |
| MONEY            | 1 milyon dollar       |
| ...              | ...                   |

*(Full list available in [model_card.md](model_card.md))*

## 🛠️ Advanced Usage

### Download Model Manually

```python
from kagglehub import model_download
model_path = model_download("afinaapayeva/ner_tuned_az/pyTorch/default")
```

### API Server

```bash
uvicorn api:app --reload
```

Then send POST requests to:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"Nobel mükafatı 2025-ci ildə təqdim olunacaq"}'
```

## 📈 Performance

| Metric     | Score |
|------------|-------|
| Precision  | 0.92  |
| Recall     | 0.89  |
| F1         | 0.90  |

![Confusion Matrix](assets/confusion_matrix.png)

## 📂 Project Structure

```
.
├── main.py             # Main CLI interface
├── ner_predictor.py    # Core model class
├── requirements.txt    # Dependencies
├── api.py              # FastAPI endpoint
└── data/
    ├── train.json      # Sample training data
    └── test.json       # Evaluation data
```

## 🤝 Contributing

1. Fork the project
2. Create your branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

## ✉️ Contact

Afina Apayeva - afina@example.com  

