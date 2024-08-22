##conversation between bot and the user

import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob


def calculate_value_matchC(conversation, values):
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform(
        [" ".join([str(msg) for msg in conversation if msg is not None])]
    )
    value_vectors = vectorizer.transform([str(value) for value in values])
    similarities = cosine_similarity(text_vector, value_vectors)
    score = similarities.sum()
    return score


def calculate_shared_interestsC(conversation, interests):
    vectorizer = TfidfVectorizer()
    text_vector = vectorizer.fit_transform(
        [" ".join([str(msg) for msg in conversation if msg is not None])]
    )
    interest_vectors = vectorizer.transform([str(interest) for interest in interests])
    similarities = cosine_similarity(text_vector, interest_vectors)
    score = similarities.sum()
    return score


def calculate_communication_style_scoreC(conversation, style):
    text = " ".join([str(msg) for msg in conversation if msg is not None]).lower()
    tokens = word_tokenize(text)
    tagged = pos_tag(tokens)
    style_words = [word for word, tag in tagged if style in tag]
    score = len(style_words)
    return score


def calculate_personality_trait_scoreC(conversation, traits):
    text = " ".join([str(msg) for msg in conversation if msg is not None]).lower()
    blob = TextBlob(text)
    score = 0
    for trait in traits:
        if trait in blob.noun_phrases:
            score += 1
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
        "attachment",
        "dedicated",
        "wholehearted",
        "trustworthiness",
        "adherence",
        "enthusiasm",
    ]
    score = 0
    for keyword in commitment_keywords:
        occurrences = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
        score += occurrences
    length = len(text.split())
    if length > 0:
        score = score / length
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
        "misunderstood",
        "grievance",
        "irritation",
        "annoyance",
        "squabble",
        "spat",
        "friction",
        "struggle",
    ]
    score = 0
    for keyword in conflict_keywords:
        occurrences = len(re.findall(r"\b" + re.escape(keyword) + r"\b", text))
        score += occurrences
    length = len(text.split())
    if length > 0:
        score = score / length
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


##conversation between the 2 users

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def get_average_embedding(text_list, model):
    embeddings = [model.wv[word] for word in text_list if word in model.wv]
    if not embeddings:
        return np.zeros(model.vector_size)
    return np.mean(embeddings, axis=0)


def calculate_similarity(user1, user2, model):
    user1_embedding = get_average_embedding(user1, model)
    user2_embedding = get_average_embedding(user2, model)
    similarity = cosine_similarity([user1_embedding], [user2_embedding])[0][0]
    return similarity * 100


def calculate_shared_interests(user1_interests, user2_interests):
    processed_sentences_1 = [sentence.lower().split() for sentence in user1_interests]
    processed_sentences_2 = [sentence.lower().split() for sentence in user2_interests]
    word2vec_model = Word2Vec(
        sentences=processed_sentences_1 + processed_sentences_2,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
    )
    score = calculate_similarity(user1_interests, user2_interests, word2vec_model)
    return score


def calculate_value_match(user1_values, user2_values):
    processed_sentences_1 = [sentence.lower().split() for sentence in user1_values]
    processed_sentences_2 = [sentence.lower().split() for sentence in user2_values]
    word2vec_model = Word2Vec(
        sentences=processed_sentences_1 + processed_sentences_2,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
    )
    score = calculate_similarity(user1_values, user2_values, word2vec_model)
    return score


def calculate_communication_style_score(user1_style, user2_style):
    processed_sentences_1 = [sentence.lower().split() for sentence in user1_style]
    processed_sentences_2 = [sentence.lower().split() for sentence in user2_style]
    word2vec_model = Word2Vec(
        sentences=processed_sentences_1 + processed_sentences_2,
        vector_size=100,
        window=1,
        min_count=1,
        sg=0,
    )
    score = calculate_similarity(user1_style, user2_style, word2vec_model)
    return score


def calculate_personality_trait_score(user1_traits, user2_traits):
    processed_sentences_1 = [sentence.lower().split() for sentence in user1_traits]
    processed_sentences_2 = [sentence.lower().split() for sentence in user2_traits]
    word2vec_model = Word2Vec(
        sentences=processed_sentences_1 + processed_sentences_2,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,
    )
    score = calculate_similarity(user1_traits, user2_traits, word2vec_model)
    return score


def calculate_mutual_commitment(user1_commitment, user2_commitment):
    processed_sentences_1 = [sentence.lower().split() for sentence in user1_commitment]
    processed_sentences_2 = [sentence.lower().split() for sentence in user2_commitment]
    word2vec_model = Word2Vec(
        sentences=processed_sentences_1 + processed_sentences_2,
        vector_size=100,
        window=1,
        min_count=1,
        sg=0,
    )
    score = calculate_similarity(user1_commitment, user2_commitment, word2vec_model)
    return score


def calculate_conflict_resolution_score(user1_resolution, user2_resolution):
    processed_sentences_1 = [sentence.lower().split() for sentence in user1_resolution]
    processed_sentences_2 = [sentence.lower().split() for sentence in user2_resolution]
    word2vec_model = Word2Vec(
        sentences=processed_sentences_1 + processed_sentences_2,
        vector_size=100,
        window=1,
        min_count=1,
        sg=0,
    )
    score = calculate_similarity(user1_resolution, user2_resolution, word2vec_model)
    return score


def rate_conversation_flow(chat_history):
    chat_history
    score = 0
    return score


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

    final_score = (
        shared_interests * weights["shared_interests"]
        + value_match * weights["value_match"]
        + comm_style * weights["comm_style"]
        + personality_traits * weights["personality_traits"]
        + mutual_commitment * weights["mutual_commitment"]
        + conflict_resolution * weights["conflict_resolution"]
        + conversation_flow * weights["conversation_flow"]
    )

    return final_score
