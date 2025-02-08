import re
import logging
import numpy as np
from fastapi import APIRouter  # type: ignore
from nltk.corpus import stopwords
from fastapi import HTTPException
from nltk.stem import WordNetLemmatizer
from src.schemas.chat import TonePredictRequest
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

router = APIRouter()

# with open("emotion.pkl", "rb") as f:
#     emotion = pickle.load(f)

lemmatizer = WordNetLemmatizer()

voc_size = 10000
sent_length = 20

emotion_dict = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise",
}


@router.post("/")
@router.post("")
async def tonepredict(data: TonePredictRequest):
    try:
        text = data.text

        text = re.sub("[^a-zA-Z]", " ", text)
        text = text.lower()
        text = text.split()
        text = [
            lemmatizer.lemmatize(word)
            for word in text
            if word not in stopwords.words("english")
        ]
        text = " ".join(text)

        onehot_repr = [one_hot(text, voc_size)]
        embedded_text = pad_sequences(onehot_repr, padding="post", maxlen=sent_length)
        X_final = np.array(embedded_text)

        y_pred = emotion.predict(X_final)
        y_pred_classes = np.argmax(y_pred, axis=1)
        sent_emotion = emotion_dict.get(y_pred_classes[0], "unknown")

        return {"emotion": sent_emotion}
    except Exception as e:
        logging.error(f"Tone predict endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during emotion prediction"
        )
