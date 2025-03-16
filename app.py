from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# ✅ Load model & tokenizer
model_path = "reazul614/sentiment-analysis-model"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# ✅ Initialize FastAPI
app = FastAPI()

# Define input format
class TextInput(BaseModel):
    text: str

# ✅ Add a Homepage Route
@app.get("/", include_in_schema=False)
@app.head("/", include_in_schema=False)
def home():
    return {"message": "🚀 Sentiment Analysis API is live! Use /predict/ to analyze text."}

# Prediction function
def predict_sentiment(text):
    try:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

        label_mapping = {0: "Negative", 1: "Neutral", 2: "Positive"}
        return {"sentiment": label_mapping[predicted_class]}
    
    except Exception as e:
        return {"error": str(e)}

@app.post("/predict/")
def get_sentiment(input_data: TextInput):
    prediction = predict_sentiment(input_data.text)
    return prediction

