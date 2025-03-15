# sentiment-analysis-api
app.py is a FastAPI-based Sentiment Analysis API that predicts the sentiment (Positive, Neutral, Negative) of a given text input. It uses a fine-tuned RoBERTa model from Hugging Face Transformers. The API has a single POST endpoint:  /predict/ → Accepts JSON input { "text": "Your sentence here" } and returns the sentiment classification.
