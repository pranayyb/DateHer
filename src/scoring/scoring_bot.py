##conversation between bot and the user

import re
import numpy as np
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

def calculate_value_matchC(conversation, values):
    try:
        conversation_embeddings = model.encode(
            [msg for msg in conversation if msg is not None]
        )
    except Exception as e:
        print(f"Error encoding conversation: {e}")
        return 0
    value_embeddings = model.encode(values)
    similarities = cosine_similarity(conversation_embeddings, value_embeddings)
    score = np.sum(np.max(similarities, axis=1))
    return score


def calculate_shared_interestsC(conversation, interests):
    conversation_text = " ".join([str(msg) for msg in conversation if msg is not None])
    conversation_embedding = model.encode([conversation_text])
    interest_embeddings = model.encode(interests)
    similarities = cosine_similarity(conversation_embedding, interest_embeddings)
    score = np.sum(similarities)
    return score


def calculate_communication_style_scoreC(conversation, style_keywords):
    text = " ".join([str(msg) for msg in conversation if msg is not None]).lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    style_words = [
        word
        for word, tag in tagged
        if any(keyword in word for keyword in style_keywords)
    ]

    score = len(style_words)
    return score


def calculate_personality_trait_scoreC(conversation, traits):
    conversation_text = " ".join([str(msg) for msg in conversation if msg is not None])
    conversation_embedding = model.encode([conversation_text])
    trait_embeddings = model.encode(traits)
    similarities = cosine_similarity(conversation_embedding, trait_embeddings)
    score = np.sum(similarities)
    return score


def calculate_mutual_commitmentC(conversation):
    text = " ".join([str(msg) for msg in conversation if msg is not None]).lower()
    commitment_keywords = [
        "commitment",
        "dedication",
        "loyalty",
        "devotion",
        "allegiance",
        "faithfulness",
        "reliability",
        "steadfastness",
        "attachment",
        "fidelity",
        "resolve",
        "pledge",
        "promise",
        "engagement",
        "consistency",
        "support",
        "responsibility",
        "dependability",
        "determination",
        "zeal",
        "persistency",
        "sincerity",
        "bond",
        "dedicated",
        "wholehearted",
        "trustworthiness",
        "adherence",
        "enthusiasm",
    ]
    score = sum(
        len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
        for keyword in commitment_keywords
    )

    length = len(text.split())
    if length > 0:
        score /= length
    return score


def calculate_conflict_resolution_scoreC(conversation):
    text = " ".join([str(msg) for msg in conversation if msg is not None]).lower()
    conflict_keywords = [
        "conflict",
        "disagreement",
        "argument",
        "debate",
        "quarrel",
        "fight",
        "strife",
        "controversy",
        "tension",
        "hostility",
        "contention",
        "clash",
        "dispute",
        "misunderstanding",
        "opposition",
        "discord",
        "issue",
        "row",
        "feud",
        "rift",
        "combat",
        "disaccord",
        "dissonance",
        "problem",
        "trouble",
        "discontent",
        "altercation",
        "grievance",
        "irritation",
        "annoyance",
        "squabble",
        "spat",
        "friction",
        "struggle",
    ]
    score = sum(
        len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
        for keyword in conflict_keywords
    )
    length = len(text.split())
    if length > 0:
        score /= length
    return score


def calculate_final_scoreC(
    shared_interests,
    value_match,
    comm_style,
    personality_traits,
    mutual_commitment,
    conflict_resolution,
):
    scores = [
        shared_interests,
        value_match,
        comm_style,
        personality_traits,
        mutual_commitment,
        conflict_resolution,
    ]
    if scores:
        max_score = max(scores)
        normalized_scores = [score / max_score for score in scores]
        total_score = sum(normalized_scores) / len(normalized_scores)
    else:
        total_score = 0
    return total_score
