# api/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Literal, Union

# ==========================================
# 共用 Input 結構
# ==========================================
class LocationInfo(BaseModel):
    location_id: str
    name: str
    address: str
    description: str
    opening_hours: Optional[str] = None
    tags: List[str]

class NpcKnowledge(BaseModel):
    expert: List[str]
    aware: List[str]
    unknown: List[str]

class NpcInfo(BaseModel):
    npc_id: str
    name: str
    role: str
    intro: str
    personality: List[str]
    speech_style: str
    preferences: Dict[str, List[str]]
    knowledge_scope: NpcKnowledge
    relationships: Dict[str, dict]
    handoff_rules: List[dict]

class NodeContext(BaseModel):
    goal: str
    scene_description: str

class DialogueHistory(BaseModel):
    speaker: str
    text: str

class DaySummary(BaseModel):
    location_id: str
    location_name: str
    summary_text: str

class Accommodation(BaseModel):
    accommodation_id: str
    name: str
    address: str
    description: str

# ==========================================
# 三大節點 Input Schemas
# ==========================================
class DialogueInput(BaseModel):
    session_id: str
    node_id: str
    node_type: Literal["dialogue"]
    model: str
    temperature: float = 0.7
    max_tokens: int = 400
    response_format: dict = {"type": "json_object"}
    location: LocationInfo
    player_preferences: List[str]
    npcs: List[NpcInfo]
    node_context: NodeContext
    dialogue_history: List[DialogueHistory] = []
    player_input: str

class OvernightTransitionInput(BaseModel):
    session_id: str
    node_id: str
    node_type: Literal["overnight_transition"]
    model: str
    temperature: float = 0.6
    max_tokens: int = 300
    response_format: dict = {"type": "json_object"}
    day_index: int
    day_summary: List[DaySummary]
    accommodation: Accommodation

class NarrationInput(BaseModel):
    session_id: str
    node_id: str
    node_type: Literal["narration"]
    model: str
    temperature: float = 0.5
    max_tokens: int = 200
    response_format: dict = {"type": "json_object"}
    day_index: int
    script_text: str

# ==========================================
# 三大節點 Output Schemas
# ==========================================
class NarrationOutput(BaseModel):
    opening_hook: str
    scene_description: str
    historical_note: Optional[str] = None

class NpcDialogueOutput(BaseModel):
    npc_id: str
    line: str
    emotion: Literal["happy", "neutral", "angry", "sad", "excited"] # 嚴格限制情緒值
    handoff_to: Optional[str] = None

class PlayerChoice(BaseModel):
    choice_id: str
    text: str

class DialogueOutput(BaseModel):
    location_id: str
    node_id: str
    narration: NarrationOutput
    npc_dialogue: List[NpcDialogueOutput]
    player_choices: List[PlayerChoice]

class OvernightTransitionOutput(BaseModel):
    day_index: int
    recap: str
    accommodation_scene: str
    next_day_hint: str

class NarrationOutputNode(BaseModel):
    day_index: int
    node_id: str
    narration_text: str
