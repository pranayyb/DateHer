from http.client import HTTPException
import json
import logging
import pickle
import random
import numpy as np
from src.chat_model.base_chat import GroqLLM
from tensorflow.keras.preprocessing.sequence import pad_sequences


label_mapping = {
    0: "apologize_after_fight",
    1: "commitment",
    2: "communication_style",
    3: "communication_tips",
    4: "conflict_resolution",
    5: "different_levels_sexual_desire",
    6: "different_priorities",
    7: "express_feelings",
    8: "family_relationships",
    9: "farewell",
    10: "feeling_insecure",
    11: "friends_dont_like_partner",
    12: "greeting",
    13: "handle_jealousy",
    14: "healthy_relationship",
    15: "hobbies",
    16: "importance_sexual_compatibility",
    17: "improve_sexual_connection",
    18: "initiate_chat",
    19: "keep_romance_alive",
    20: "manage_long_distance",
    21: "matching_preferences",
    22: "photo_advice",
    23: "planning",
    24: "profile_creation",
    25: "safety_advice",
    26: "small_talk",
    27: "social_behavior",
    28: "spark_fading",
    29: "support_partner_through_tough_times",
    30: "talk_about_sex",
    31: "values",
}


# with open("tokenizer.pickle", "rb") as handle:
#     tokenizer = pickle.load(handle)

model_name = "mixtral-8x7b-32768"
model = GroqLLM(api_key=GROQ_API_KEY, model_name=model_name)

with open("data/intents.json") as f:
    data = json.load(f)


def predict(text):
    try:
        sequences = tokenizer.texts_to_sequences([text])
        if not sequences or not sequences[0]:
            raise ValueError(
                "Input text resulted in an empty sequence. Please provide valid input."
            )
        padded_sequences = pad_sequences(sequences, maxlen=16)
        pred = model.predict(padded_sequences)
        prediction = np.argmax(pred)
        predicted_label = label_mapping[prediction]
        return predicted_label
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Error in prediction")


def answer(text):
    try:
        predicted_label = predict(text)
        for i in range(len(data["intents"])):
            if data["intents"][i]["tag"] == predicted_label:
                return random.choice(data["intents"][i]["responses"])
        return "Sorry, I didn't understand that."
    except Exception as e:
        logging.error(f"Error in generating answer: {e}")
        return "Sorry, I couldn't generate an answer."
