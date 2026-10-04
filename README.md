# Human vs AI Writing Pattern Analyzer

An NLP project that extracts linguistic and stylistic features from a text (sentence variation, vocabulary diversity, repetition, burstiness, formality, paragraph structure, readability, POS patterns) and uses a machine-learning classifier to predict whether the writing is closer to human-written or AI-generated text. It explains its prediction and can compare two texts.

> The result is a prediction based on style patterns in the training data, not proof of authorship.

## Structure

```
├── app.py                    # Streamlit app
├── features.py               # NLP feature extraction
├── writing_analyzer.joblib   # trained model (from the Colab notebook)
├── requirements.txt
├── notebooks/                # training notebook (Colab)
└── .streamlit/config.toml    # theme
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate        # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Model

Trained on the Kaggle [AI vs Human Text](https://www.kaggle.com/datasets/shanegerami/ai-vs-human-text) dataset plus Wikipedia/GPT intro pairs. Models compared: Logistic Regression, Random Forest, XGBoost.

## Limitations

- Formal human writing (research papers, encyclopedias) can be flagged as AI-like.
- The model only knows the text types and generators it was trained on.
- Input must be at least 100 words and 3 sentences.

