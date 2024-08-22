from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import tensorflow
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
import random
import json
import re
import pandas as pd
import httpx
import datetime
import logging
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS


with open("intents.json") as f:
    data = json.load(f)


model = tensorflow.keras.models.load_model(r"chatbot_model.h5")
lemmatizer = WordNetLemmatizer()

from scoring import (
    calculate_shared_interestsC,
    calculate_value_matchC,
    calculate_communication_style_scoreC,
    calculate_personality_trait_scoreC,
    calculate_mutual_commitmentC,
    calculate_conflict_resolution_scoreC,
    calculate_final_scoreC,
    calculate_shared_interests,
    calculate_value_match,
    calculate_communication_style_score,
    calculate_personality_trait_score,
    calculate_mutual_commitment,
    calculate_conflict_resolution_score,
    calculate_final_score,
    rate_conversation_flow,
)

emotion_dict = {0: "sadness", 1: "joy", 2: "love", 3: "anger", 4: "fear", 5: "surprise"}

voc_size = 10000
sent_length = 20

CONVERSATION_THRESHOLD = 50

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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

with open("tokenizer.pickle", "rb") as handle:
    tokenizer = pickle.load(handle)

with open("emotion.pkl", "rb") as f:
    emotion = pickle.load(f)

def predict(text):
    pred = model.predict(pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=16))
    prediction = np.argmax(pred)
    predicted_label = label_mapping[prediction]
    return predicted_label

def answer(text):
    predicted_label = predict(text)
    for i in range(len(data["intents"])):
        if data["intents"][i]["tag"] == predicted_label:
            return random.choice(data["intents"][i]["responses"])


user_data = {}

questions = [
    "Could you share some of your hobbies and interests?",
    "Which values do you hold in high regard?",
    "How significant are family and friends in your life?",
    "How would you characterize your social interactions?",
    "In your view, what constitutes a healthy relationship?",
    "What is your understanding of commitment?",
    "Do you prefer to plan ahead or are you more spontaneous?",
    "Would you say you're more inclined towards phone calls or texting?",
]

responses = {
    "intro": "Heyyy!!....Who doesn’t need love? We all do, and that’s exactly why we’re here—to help you find what you’re searching for. ",
    "greet": "I hope you're having a wonderful day. We're thrilled that you've chosen us to help you find your perfect match! To get to know you better and speed up the process of finding your soulmate, may we ask you a few questions? 😉",
    "normal_conversation": "Let's talk about relationships. What are your thoughts?",
    "ask_for_questions": "I'd like to ask you a few questions to get to know you better. Are you okay with that?",
    "thank_you": "Thank you for answering all the questions! If you have any more questions about dating, feel free to ask.",
    "farewell": "Goodbye! Have a great day!",
    "new": "TO start a new conversation enter start!",
}


class ChatRequest(BaseModel):
    user_id: str
    message: Optional[str] = None


class UserMessage(BaseModel):
    text: str
    createdAt: dict


class CalcRequest(BaseModel):
    user1: dict
    user2: dict
    message: List[UserMessage]


class MatchRequest(BaseModel):
    user_id: str


class TonePredictRequest(BaseModel):
    text: str


@app.post("/chat")
async def chat(data: ChatRequest):
    comp = 0
    user_id = data.user_id
    user_message = data.message

    affirmative_words = [
        "yes",
        "okay",
        "sure",
        "absolutely",
        "of course",
        "definitely",
        "alright",
        "yeah",
        "yep",
        "certainly",
        "indeed",
    ]

    if user_id not in user_data:
        user_data[user_id] = {
            "conversation": [],
            "current_question_index": 0,
            "interests": [],
            "values": [],
            "style": ["commitment"],
            "traits": [],
            "state": "intro",
        }

    user_info = user_data[user_id]
    conversation = user_info["conversation"]
    current_question_index = user_info["current_question_index"]
    state = user_info["state"]

    # print(current_question_index)
    # print(user_message)
    # if user_message==None and current_question_index!=0:
    #     return "Enter a message first!"

    if state == "intro":
        response = responses["intro"]
        user_info["state"] = "greet"
    elif state == "greet":
        response = responses["greet"]
        user_info["state"] = "ask_for_questions"
    elif state == "ask_for_questions":
        if any(word in user_message.lower() for word in affirmative_words):
            response = questions[current_question_index]
            user_info["state"] = "asking_questions"
        else:
            response = "Alright, let me know if you change your mind."
            user_info["state"] = "ask_for_questions"
    elif state == "asking_questions":
        if current_question_index < len(questions):
            if current_question_index == 0:
                user_info["interests"].extend(
                    [i.strip() for i in user_message.split(",")]
                )
            elif current_question_index == 1:
                user_info["values"].extend([v.strip() for v in user_message.split(",")])
            elif current_question_index == 6:
                user_info["traits"].extend([t.strip() for t in user_message.split(",")])
            elif current_question_index == 7:
                user_info["style"] = user_message.strip()

            current_question_index += 1
            user_info["current_question_index"] = current_question_index

            if current_question_index < len(questions):
                response = questions[current_question_index]
            else:
                shared_interests = calculate_shared_interestsC(
                    conversation, user_info["interests"]
                )
                value_match = calculate_value_matchC(conversation, user_info["values"])
                comm_style = calculate_communication_style_scoreC(
                    user_info["style"], "text"
                )
                personality_traits = calculate_personality_trait_scoreC(
                    conversation, user_info["traits"]
                )
                mutual_commitment = calculate_mutual_commitmentC(conversation)
                conflict_resolution = calculate_conflict_resolution_scoreC(conversation)

                final_score = calculate_final_scoreC(
                    shared_interests,
                    value_match,
                    comm_style,
                    personality_traits,
                    mutual_commitment,
                    conflict_resolution,
                )

                comp = f"{final_score:.2f}"
                user_info["state"] = "thank_you"
                response = responses["thank_you"]
                user_info["state"] = "awaiting_questions"
    elif state == "awaiting_questions":
        response = answer(user_message)
        predicted_label = predict(user_message)

        if predicted_label == "farewell":
            user_info["state"] = "farewell"
        else:
            user_info["state"] = "awaiting_questions"

    elif state == "farewell":
        response = responses["farewell"]
        user_info["state"] = "new"

    elif state == "new":
        response = answer(user_message)
        predicted_label = predict(user_message)
        user_info["state"] = "awaiting_questions"

    user_info["conversation"].append(user_message)

    response_data = {
        "response": response,
        "current_question_index": user_info["current_question_index"],
        "conversation": user_info["conversation"],
        "user_info": user_info,
    }

    if comp != 0:
        response_data["compatibility"] = comp

    return response_data


@app.post("/calc")
async def calc(data: CalcRequest):
    comp_score = 0
    chat_history = [msg.text for msg in data.message]

    if len(chat_history) > CONVERSATION_THRESHOLD:
        user1_data = {
            "interests": data.user1["interests"],
            "values": data.user1["values"],
            "style": data.user1["style"],
            "traits": data.user1["traits"],
            "commitment": data.user1["commitment"],
            "resolution": data.user1["resolution"],
        }
        user2_data = {
            "interests": data.user2["interests"],
            "values": data.user2["values"],
            "style": data.user2["style"],
            "traits": data.user2["traits"],
            "commitment": data.user2["commitment"],
            "resolution": data.user2["resolution"],
        }

        shared_interests = calculate_shared_interests(
            user1_data["interests"], user2_data["interests"]
        )
        value_match = calculate_value_match(user1_data["values"], user2_data["values"])
        comm_style = calculate_communication_style_score(
            user1_data["style"], user2_data["style"]
        )
        personality_traits = calculate_personality_trait_score(
            user1_data["traits"], user2_data["traits"]
        )
        mutual_commitment = calculate_mutual_commitment(
            user1_data["commitment"], user2_data["commitment"]
        )
        conflict_resolution = calculate_conflict_resolution_score(
            user1_data["resolution"], user2_data["resolution"]
        )
        conversation_flow = rate_conversation_flow(chat_history)

        final_score = calculate_final_score(
            shared_interests,
            value_match,
            comm_style,
            personality_traits,
            mutual_commitment,
            conflict_resolution,
            conversation_flow,
        )

        comp_score = f"{final_score:.2f}"
    else:
        comp_score = (
            "Conversation threshold not reached yet to calculate compatibility score!."
        )

    return {"comp_score": comp_score}


@app.post("/match")
async def match(data: MatchRequest):

    async with httpx.AsyncClient() as client:
        response = await client.get(
            "http://ec2-3-7-69-234.ap-south-1.compute.amazonaws.com:3001/getboys"
        )
    boys_data = response.json()
    # print(boys_data)
    # print(response)
    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail="Failed to retrieve data from external API",
        )

    users = pd.DataFrame(boys_data)
    # print(users)
    # print(users.shape)
    # print(users.columns)

    user_id = int(data.user_id)
    # print(type(user_id))
    # print(type(users["id"]))

    current_user = users[users["id"] == user_id]
    if current_user.empty:
        raise HTTPException(status_code=404, detail="User not found")

    current_user = current_user.iloc[0]

    age_min = int(current_user["age"]) - 2
    age_max = int(current_user["age"]) + 2

    potential_matches = users[
        (users["age"] >= age_min) & (users["age"] <= age_max) & (users["id"] != user_id)
    ]

    if potential_matches.empty:
        return []

    all_interests = potential_matches["interests"].tolist() + [
        current_user["interests"]
    ]

    vectorizer = CountVectorizer().fit(all_interests)
    all_vectors = vectorizer.transform(all_interests).toarray()

    current_user_vector = all_vectors[-1]
    potential_matches_vectors = all_vectors[:-1]

    similarity = cosine_similarity(
        [current_user_vector], potential_matches_vectors
    ).flatten()

    potential_matches["similarity"] = similarity

    potential_matches = potential_matches.sort_values(by="similarity", ascending=False)

    top_matches = potential_matches.head()

    result = top_matches[
        ["id", "first_name", "last_name", "age", "location", "interests", "similarity"]
    ].to_dict(orient="records")

    return result


@app.post("/tonepredict")
async def tonepredict(data: TonePredictRequest):
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
