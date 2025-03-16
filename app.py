from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# ✅ Load model & tokenizer
model_path = "/content/drive/MyDrive/saved_model"
model = RobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# ✅ Initialize FastAPI
app = FastAPI()

# Define input format
class TextInput(BaseModel):
    text: str

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

from transformers import RobertaForSequenceClassification, RobertaTokenizer

# ✅ Load model & tokenizer from Hugging Face
MODEL_PATH = "reazul614/sentiment-analysis-model"

model = RobertaForSequenceClassification.from_pretrained(MODEL_PATH)
tokenizer = RobertaTokenizer.from_pretrained(MODEL_PATH)

print(f"✅ Model loaded successfully from Hugging Face: {MODEL_PATH}")
