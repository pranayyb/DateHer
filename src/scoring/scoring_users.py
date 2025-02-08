##conversation between the 2 users
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def get_sentence_embedding(text_list):
    sentences = [sentence.lower().strip() for sentence in text_list]
    return model.encode(sentences).mean(axis=0)


def calculate_similarity(text1, text2):
    embedding1 = get_sentence_embedding(text1)
    embedding2 = get_sentence_embedding(text2)
    return cosine_similarity([embedding1], [embedding2])[0][0] * 100


def calculate_shared_interests(user1_interests, user2_interests):
    return calculate_similarity(user1_interests, user2_interests)


def calculate_value_match(user1_values, user2_values):
    return calculate_similarity(user1_values, user2_values)


def calculate_communication_style_score(user1_style, user2_style):
    return calculate_similarity(user1_style, user2_style)


def calculate_personality_trait_score(user1_traits, user2_traits):
    return calculate_similarity(user1_traits, user2_traits)


def calculate_mutual_commitment(user1_commitment, user2_commitment):
    return calculate_similarity(user1_commitment, user2_commitment)


def calculate_conflict_resolution_score(user1_resolution, user2_resolution):
    return calculate_similarity(user1_resolution, user2_resolution)


def rate_conversation_flow(chat_history):
    return 1


def calculate_final_score(
    shared_interests,
    value_match,
    comm_style,
    personality_traits,
    mutual_commitment,
    conflict_resolution,
    conversation_flow,
):
    weights = {
        "shared_interests": 0.25,
        "value_match": 0.20,
        "comm_style": 0.20,
        "personality_traits": 0.10,
        "mutual_commitment": 0.05,
        "conflict_resolution": 0.05,
        "conversation_flow": 0.15,
    }
    return sum(
        metric * weights[key]
        for key, metric in zip(
            weights.keys(),
            [
                shared_interests,
                value_match,
                comm_style,
                personality_traits,
                mutual_commitment,
                conflict_resolution,
                conversation_flow,
            ],
        )
    )
