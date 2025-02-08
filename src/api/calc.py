import asyncio
import httpx
import logging
from fastapi import HTTPException
from src.schemas.chat import CalcRequest
from fastapi import APIRouter  # type: ignore
from src.scoring.scoring_users import (
    calculate_communication_style_score,
    calculate_conflict_resolution_score,
    calculate_final_score,
    calculate_mutual_commitment,
    calculate_personality_trait_score,
    calculate_shared_interests,
    calculate_value_match,
    rate_conversation_flow,
)

router = APIRouter()

CONVERSATION_THRESHOLD = 50


@router.post("/")
@router.post("")
async def calc(request: CalcRequest):
    try:
        comp_score = 0
        chat_history = [entry["text"] for entry in request.data]
        sender_ids = {entry["senderId"] for entry in request.data}
        sender_ids = list(sender_ids)
        async with httpx.AsyncClient() as client:
            response1 = await asyncio.gather(
                client.get(
                    "http://ec2-3-7-69-234.ap-south-1.compute.amazonaws.com:3001/getallusers"
                )
            )
            response1 = response1[0]
            if response1.status_code != 200:
                raise HTTPException(
                    status_code=response1.status_code,
                    detail="Failed to retrieve users data from external API",
                )
            users_data = response1.json()

        users_dict = {user["id"]: user for user in users_data}
        sender_ids_int = [int(id_str) for id_str in sender_ids]
        user_data = {user_id: users_dict.get(user_id) for user_id in sender_ids_int}
        user_data_list = list(user_data.values())
        user1_data = user_data_list[0]
        user2_data = user_data_list[1]

        if len(chat_history) > CONVERSATION_THRESHOLD:
            user1_data = {
                "interests": user1_data.get("interests", []),
                "values": user1_data.get("values", []),
                "style": user1_data.get("style", ""),
                "traits": user1_data.get("traits", []),
                "commitment": user1_data.get("commitment", []),
                "resolution": user1_data.get("resolution", []),
            }
            user2_data = {
                "interests": user2_data.get("interests", []),
                "values": user2_data.get("values", []),
                "style": user2_data.get("style", ""),
                "traits": user2_data.get("traits", []),
                "commitment": user2_data.get("commitment", []),
                "resolution": user2_data.get("resolution", []),
            }
            print(user1_data, user2_data)
            # Calculate different scores
            shared_interests = calculate_shared_interests(
                user1_data["interests"], user2_data["interests"]
            )
            value_match = calculate_value_match(
                user1_data["values"], user2_data["values"]
            )
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
            comp_score = "Conversation threshold not reached yet to calculate compatibility score!"

        return {"compatibility_score": comp_score}
    except Exception as e:
        logging.error(f"Calc endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during compatibility calculation"
        )
