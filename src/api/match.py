import asyncio
from fastapi import HTTPException
import logging
import httpx
import pandas as pd
from src.schemas.chat import MatchRequest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter  # type: ignore

router = APIRouter()


@router.post("/")
@router.post("")
async def match(data: MatchRequest):
    try:
        async with httpx.AsyncClient() as client:
            response1, response2 = await asyncio.gather(
                client.get(
                    "http://ec2-3-7-69-234.ap-south-1.compute.amazonaws.com:3001/getboys"
                ),
                client.get(
                    "http://ec2-3-7-69-234.ap-south-1.compute.amazonaws.com:3001/getgirls"
                ),
            )

            if response1.status_code != 200:
                raise HTTPException(
                    status_code=response1.status_code,
                    detail="Failed to retrieve boys data from external API",
                )
            if response2.status_code != 200:
                raise HTTPException(
                    status_code=response2.status_code,
                    detail="Failed to retrieve girls data from external API",
                )

            boys_data = response1.json()
            girls_data = response2.json()

        girls_df = pd.DataFrame(girls_data)
        boys_df = pd.DataFrame(boys_data)

        user_id = int(data.user_id)
        current_user = girls_df[girls_df["id"] == user_id]

        if current_user.empty:
            raise HTTPException(status_code=404, detail="User not found")

        current_user = current_user.iloc[0]
        age_min = int(current_user["age"]) - 2
        age_max = int(current_user["age"]) + 2

        potential_matches = boys_df[
            (boys_df["age"] >= age_min)
            & (boys_df["age"] <= age_max)
            & (boys_df["id"] != user_id)
        ]

        if potential_matches.empty:
            return []

        potential_matches["interests"] = potential_matches["interests"].fillna("")
        current_user_interests = current_user["interests"] or ""

        all_interests = potential_matches["interests"].tolist() + [
            current_user_interests
        ]
        vectorizer = CountVectorizer().fit(all_interests)
        all_vectors = vectorizer.transform(all_interests).toarray()

        current_user_vector = all_vectors[-1]
        potential_matches_vectors = all_vectors[:-1]

        similarity = cosine_similarity(
            [current_user_vector], potential_matches_vectors
        ).flatten()

        potential_matches["similarity"] = similarity
        potential_matches = potential_matches.sort_values(
            by="similarity", ascending=False
        )

        result = potential_matches.to_dict(orient="records")

        return result
    except Exception as e:
        logging.error(f"Match endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during the match process"
        )
