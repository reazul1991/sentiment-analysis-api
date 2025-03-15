%%writefile README.md
# 🚀 Sentiment Analysis API - FastAPI

This is a **Sentiment Analysis API** built using **FastAPI** and **Hugging Face Transformers**.  
It predicts whether a given text is **Positive, Neutral, or Negative**.

## 📌 Features
- Uses a **fine-tuned RoBERTa model** for sentiment classification.
- Deployed using **FastAPI** for quick and scalable predictions.
- Public API endpoint available (if deployed).

## 🔹 API Endpoints
| Method | Endpoint      | Description |
|--------|-------------|-------------|
| `POST` | `/predict/` | Predict sentiment for given text |

### 🔹 Example Request
```python
import requests

url = "https://YOUR_DEPLOYED_API_URL/predict/"
response = requests.post(url, json={"text": "I love this product!"})
print(response.json())  # {"sentiment": "Positive"}
