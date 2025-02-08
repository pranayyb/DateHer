import os
import uvicorn
import nltk
from fastapi import FastAPI
from src.api import chat, calc, match, tonepredict
from fastapi.middleware.cors import CORSMiddleware

MIN_CORPORA = [
    "brown",  # Required for FastNPExtractor
    "punkt",  # Required for WordTokenizer
    "wordnet",  # Required for lemmatization
    "averaged_perceptron_tagger",  # Required for NLTKTagger
]
ADDITIONAL_CORPORA = [
    "conll2000",  # Required for ConllExtractor
    "movie_reviews",  # Required for NaiveBayesAnalyzer
]

ALL_CORPORA = MIN_CORPORA + ADDITIONAL_CORPORA
for each in ALL_CORPORA:
    nltk.download(each)
nltk.download("punkt_tab")
nltk.download("averaged_perceptron_tagger_eng")


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.include_router(chat.router, prefix="/chat", tags=["chat"])
app.include_router(calc.router, prefix="/calc", tags=["calc"])
app.include_router(match.router, prefix="/match", tags=["match"])
app.include_router(tonepredict.router, prefix="/tonepredict", tags=["tonepredict"])


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9000)
