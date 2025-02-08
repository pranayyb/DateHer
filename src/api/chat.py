from fastapi import HTTPException
import logging
from src.chat_model.base_chat import char, gen
from src.schemas.chat import ChatRequest
from src.scoring.scoring_bot import (
    calculate_communication_style_scoreC,
    calculate_conflict_resolution_scoreC,
    calculate_final_scoreC,
    calculate_mutual_commitmentC,
    calculate_personality_trait_scoreC,
    calculate_shared_interestsC,
    calculate_value_matchC,
)
from fastapi import APIRouter  # type: ignore

router = APIRouter()

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


@router.post("/")
@router.post("")
async def chat(data: ChatRequest):
    try:
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
                "conversation": [""],
                "chat_history": [""],
                "current_question_index": 0,
                "interests": [""],
                "values": [""],
                "style": ["i"],
                "traits": [""],
                "state": "intro",
                "commitment": [""],
                "resolution": [""],
            }
        chat_history = []
        response = ""
        user_info = user_data[user_id]
        conversation = user_info["conversation"]
        chat_history = user_info["chat_history"]
        current_question_index = user_info["current_question_index"]
        state = user_info["state"]

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
                    lts = char(user_message, "interests")
                    user_info["interests"] = lts if lts is not None else []
                elif current_question_index == 1:
                    lts = char(user_message, "values")
                    user_info["values"] = user_info.get("values", [])
                    user_info["values"].extend(lts if lts is not None else [])
                elif current_question_index == 2:
                    lts = char(user_message, "traits")
                    user_info["traits"] = user_info.get("traits", [])
                    user_info["traits"].extend(lts if lts is not None else [])
                elif current_question_index == 3:
                    lts = char(user_message, "traits")
                    user_info["traits"] = user_info.get("traits", [])
                    user_info["traits"].extend(lts if lts is not None else [])
                elif current_question_index == 4:
                    lts = char(user_message, "commitment")
                    user_info["commitment"] = user_info.get("commitment", [])
                    user_info["commitment"].extend(lts if lts is not None else [])
                elif current_question_index == 5:
                    lts = char(user_message, "resolution")
                    user_info["resolution"] = user_info.get("resolution", [])
                    user_info["resolution"].extend(lts if lts is not None else [])
                elif current_question_index == 6:
                    lts = char(user_message, "resolution")
                    user_info["resolution"] = user_info.get("resolution", [])
                    user_info["resolution"].extend(lts if lts is not None else [])
                elif current_question_index == 7:
                    lts = char(user_message, "style")
                    user_info["style"] = user_info.get("style", [])
                    user_info["style"].extend(lts if lts is not None else [])

                current_question_index += 1
                user_info["current_question_index"] = current_question_index

                if current_question_index < len(questions):
                    response = questions[current_question_index]
                else:
                    shared_interests = calculate_shared_interestsC(
                        conversation, user_info["interests"]
                    )
                    value_match = calculate_value_matchC(
                        conversation, user_info["values"]
                    )
                    comm_style = calculate_communication_style_scoreC(
                        user_info["style"], "text"
                    )
                    personality_traits = calculate_personality_trait_scoreC(
                        conversation, user_info["traits"]
                    )
                    mutual_commitment = calculate_mutual_commitmentC(conversation)
                    conflict_resolution = calculate_conflict_resolution_scoreC(
                        conversation
                    )

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
                    # response = gen(user_message)
                    response = gen(user_message, chat_history)
                    user_info["state"] = "awaiting_questions"
        elif state == "awaiting_questions":
            # response = gen(user_message)
            response = gen(user_message, chat_history)
        elif state == "new":
            # response = gen(user_message)
            response = gen(user_message, chat_history)
            user_info["state"] = "awaiting_questions"

        # user_info["conversation"].append(user_message)
        user_info["conversation"].append(user_message)
        user_info["chat_history"].append({"role": "user", "content": user_message})
        user_info["chat_history"].append({"role": "assistant", "content": response})

        # implement this after link of aws is corrected!
        # if current_question_index == 8:
        #     urlput = "http://ec2-3-7-69-234.ap-south-1.compute.amazonaws.com:3001/updatecharacter"
        #     datafinal = {
        #         "email": user_info.get("email", ""),
        #         "interests": user_info.get("interests", []),
        #         "values": user_info.get("values", []),
        #         "style": user_info.get("style", ""),
        #         "traits": user_info.get("traits", []),
        #         "commitment": user_info.get("commitment", []),
        #         "resolution": user_info.get("resolution", []),
        #     }
        #     try:
        #         requests.put(urlput, json=datafinal)
        #     except requests.RequestException as e:
        #         logging.error(f"Error updating user character: {e}")

        response_data = {
            "response": response,
            "current_question_index": user_info["current_question_index"],
            "conversation": user_info["conversation"],
            "user_info": user_info,
            "chat_history": user_info["chat_history"],
        }

        if comp != 0:
            response_data["compatibility"] = comp

        return response_data
    except Exception as e:
        logging.error(f"Chat endpoint error: {e}")
        raise HTTPException(
            status_code=500, detail="An error occurred during chat processing"
        )
