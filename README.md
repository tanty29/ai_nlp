# 📊 Sentiment Analysis Prototype (NLP)

End-to-end sentiment analysis of customer reviews using TF–IDF + Naïve Bayes / Linear SVM / KNN. 
Includes Jupyter training notebook, a professional Streamlit app for single & batch predictions, and evaluation artifacts.

## Quickstart
```bash
pip install -r requirements.txt
# Train in Notebook: open Sentiment_Project_Notebook.ipynb
# Or, if you already trained, ensure you have:
# models/best_model_LinearSVM.joblib and outputs/metrics_summary.csv + summary.json
streamlit run app_streamlit.py
```

## Structure
- `Sentiment_Project_Notebook.ipynb` — training, evaluation, confusion matrices, explainability, save model
- `app_streamlit.py` — Streamlit app (single + batch predictions, metrics, optional explainability)
- `models/` — saved best model (`.joblib`)
- `outputs/` — metrics & plots
- `requirements.txt` — dependencies
- `sample_reviews.csv` — small batch test file
