from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence

# Constants
VOCAB_SIZE = 10000
MAX_LEN = 500

# Load word index and model
word_index = imdb.get_word_index()
model = load_model('8-Deep Learning/Simple RNN/imdb_rnn_model.keras', compile=False)

# FastAPI app
app = FastAPI()

# JSON schema
class ReviewRequest(BaseModel):
    review: str

def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    encoded = [i for i in encoded if i < VOCAB_SIZE]
    return sequence.pad_sequences([encoded], maxlen=MAX_LEN)

@app.post("/predict")
def predict_sentiment(req: ReviewRequest):
    review = req.review.strip()
    if len(review.split()) < 25:
        raise HTTPException(status_code=400, detail="Please enter at least 25 words.")
    
    try:
        input_data = preprocess_text(review)
        prediction = model.predict(input_data)
        score = float(prediction[0][0])
        sentiment = "Positive" if score > 0.5 else "Negative"
        return {"sentiment": sentiment, "score": round(score, 4)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")