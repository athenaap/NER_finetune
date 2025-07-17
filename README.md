# Azerbaijani Named Entity Recognition (NER) Model

Research:

Named Entity Recognition (NER) is a subfield of computer science and Natural Language Processing (NLP) that focuses on identifying and classifying entities in unstructured text into predefined categories, such as persons, geographical locations, and organizations.
This information helps to analyze and process texts, giving capability to derive meaning from sentences. NER is utilized in lots of domains such as media, search engines and content recommendations. 

Historically, early NER systems relied on rule-based approaches with hand-crafted rules, lexicons, and spelling features. These methods, while simple and interpretable, lacked flexibility and scalability. The introduction of machine learning techniques marked a significant shift in the field, allowing more adaptable and data-driven approaches. With the rise of neural networks, NER systems further improved, particularly with the adoption of deep learning methods, which enabled more sophisticated models capable of capturing complex patterns in text. Most recently, Transformer-based architectures have set new standards in NER performance, leading to breakthroughs in the field. 

Another notable gap in the literature is the limited attention paid to methods that address low-resource settings, where annotated data is scarce. Producing annotated datasets is often expensive and time consuming, making it essential to develop methods that can perform effectively with limited data.

Here I present a pre-trained model for recognizing named entities in Azerbaijani text, supporting 25+ entity types including persons, locations, organizations, dates, and more.

Dataset that was used in this task ---> https://huggingface.co/datasets/LocalDoc/azerbaijani-ner-dataset 

## Installation

```bash
# Clone the repository
git clone https://github.com/athenaap/NER_finetune
cd NER_finetune

# Install dependencies
pip install -r requirements.txt
```


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

## Supported Entity Types

0: O: Outside any named entity
1: PERSON: Names of individuals
2: LOCATION: Geographical locations, both man-made and natural
3: ORGANISATION: Names of companies, institutions
4: DATE: Dates or periods
5: TIME: Times of the day
6: MONEY: Monetary values
7: PERCENTAGE: Percentage values
8: FACILITY: Buildings, airports, etc.
9: PRODUCT: Products and goods
10: EVENT: Events and occurrences
11: ART: Artworks, titles of books, songs
12: LAW: Legal documents
13: LANGUAGE: Languages
14: GPE: Countries, cities, states
15: NORP: Nationalities or religious or political groups
16: ORDINAL: Ordinal numbers
17: CARDINAL: Cardinal numbers
18: DISEASE: Diseases and medical conditions
19: CONTACT: Contact information, e.g., phone numbers, emails
20: ADAGE: Proverbs, sayings
21: QUANTITY: Measurements and quantities
22: MISCELLANEOUS: Miscellaneous entities
23: POSITION: Professional or social positions
24: PROJECT: Names of projects or programs


### Download Model Manually

```python
from kagglehub import model_download
model_path = model_download("afinaapayeva/ner_tuned_az/pyTorch/default")
```




