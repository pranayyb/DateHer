##conversation between bot and the user

import re
import nltk
from nltk import pos_tag
from nltk.tokenize import word_tokenize

# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
# from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import pos_tag, word_tokenize
import re
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_value_matchC(conversation, values):
    # model = SentenceTransformer("all-MiniLM-L6-v2")
    try:
        conversation_embeddings = model.encode([msg for msg in conversation if msg is not None])
    except Exception as e:
        print(f"Error encoding conversation: {e}")
        return 0  # Or handle it appropriately
    # conversation_embeddings = model.encode(
    #     [msg for msg in conversation if msg is not None]
    # )
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


##conversation between the 2 users

# from gensim.models import Word2Vec
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np


# def initialize_word2vec(sentences, vector_size=100, window=5, min_count=1, sg=0):
#     return Word2Vec(
#         sentences=sentences,
#         vector_size=vector_size,
#         window=window,
#         min_count=min_count,
#         sg=sg,
#     )


# def get_average_embedding(text_list, model):
#     embeddings = [model.wv[word] for word in text_list if word in model.wv]
#     if not embeddings:
#         return np.zeros(model.vector_size)
#     return np.mean(embeddings, axis=0)


# def calculate_similarity(user1, user2, model):
#     user1_embedding = get_average_embedding(user1, model)
#     user2_embedding = get_average_embedding(user2, model)
#     similarity = cosine_similarity([user1_embedding], [user2_embedding])[0][0]
#     return similarity * 100


# def calculate_shared_interests(user1_interests, user2_interests):
#     processed_sentences = [
#         sentence.lower().split() for sentence in user1_interests + user2_interests
#     ]
#     model = initialize_word2vec(processed_sentences)
#     return calculate_similarity(user1_interests, user2_interests, model)


# def calculate_value_match(user1_values, user2_values):
#     processed_sentences = [
#         sentence.lower().split() for sentence in user1_values + user2_values
#     ]
#     model = initialize_word2vec(processed_sentences)
#     return calculate_similarity(user1_values, user2_values, model)


# def calculate_communication_style_score(user1_style, user2_style):
#     processed_sentences = [
#         sentence.lower().split() for sentence in user1_style + user2_style
#     ]
#     model = initialize_word2vec(processed_sentences, window=1)
#     return calculate_similarity(user1_style, user2_style, model)


# def calculate_personality_trait_score(user1_traits, user2_traits):
#     processed_sentences = [
#         sentence.lower().split() for sentence in user1_traits + user2_traits
#     ]
#     model = initialize_word2vec(processed_sentences)
#     return calculate_similarity(user1_traits, user2_traits, model)


# def calculate_mutual_commitment(user1_commitment, user2_commitment):
#     processed_sentences = [
#         sentence.lower().split() for sentence in user1_commitment + user2_commitment
#     ]
#     model = initialize_word2vec(processed_sentences, window=1)
#     return calculate_similarity(user1_commitment, user2_commitment, model)


# def calculate_conflict_resolution_score(user1_resolution, user2_resolution):
#     processed_sentences = [
#         sentence.lower().split() for sentence in user1_resolution + user2_resolution
#     ]
#     model = initialize_word2vec(processed_sentences, window=1)
#     return calculate_similarity(user1_resolution, user2_resolution, model)


# def rate_conversation_flow(chat_history):
#     return 0


# def calculate_final_score(
#     shared_interests,
#     value_match,
#     comm_style,
#     personality_traits,
#     mutual_commitment,
#     conflict_resolution,
#     conversation_flow,
# ):
#     weights = {
#         "shared_interests": 0.25,
#         "value_match": 0.20,
#         "comm_style": 0.20,
#         "personality_traits": 0.10,
#         "mutual_commitment": 0.05,
#         "conflict_resolution": 0.05,
#         "conversation_flow": 0.15,
#     }
#     final_score = (
#         shared_interests * weights["shared_interests"]
#         + value_match * weights["value_match"]
#         + comm_style * weights["comm_style"]
#         + personality_traits * weights["personality_traits"]
#         + mutual_commitment * weights["mutual_commitment"]
#         + conflict_resolution * weights["conflict_resolution"]
#         + conversation_flow * weights["conversation_flow"]
#     )
#     return final_score


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# model = SentenceTransformer("all-MiniLM-L6-v2")


def get_sentence_embedding(text_list):
    sentences = [" ".join(sentence.lower().split()) for sentence in text_list]
    return model.encode(sentences).mean(axis=0)


def calculate_similarity(user1, user2):
    user1_embedding = get_sentence_embedding(user1)
    user2_embedding = get_sentence_embedding(user2)
    similarity = cosine_similarity([user1_embedding], [user2_embedding])[0][0]
    return similarity * 100


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
    return 0


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
