from pydantic import BaseModel
from typing import Optional, List, Dict


class ChatRequest(BaseModel):
    user_id: str
    message: Optional[str] = None


class UserMessage(BaseModel):
    text: str
    createdAt: dict


class CalcRequest(BaseModel):
    data: List[Dict]


class MatchRequest(BaseModel):
    user_id: str


class TonePredictRequest(BaseModel):
    text: str
