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

# ==========================================
# 劇本任務批次生成 Schemas
# ==========================================
class TaskTypeItem(BaseModel):
    type_id: int
    type_name: str

class PlaceItem(BaseModel):
    place_id: str
    p_name: str
    is_hotel: int
    is_hidden: int
    type_list: List[TaskTypeItem]

class StoryCondition(BaseModel):
    story_no: int
    nt_name: str
    places: List[PlaceItem]

class StoryTaskRequest(BaseModel):
    city_name: str
    district_name: str
    party_size: int
    s_tag: List[str]
    is_night_mode: int
    stories: List[StoryCondition]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "city_name": "臺中市",
                    "district_name": "西區",
                    "party_size": 2,
                    "s_tag": ["好山好水", "美食"],
                    "is_night_mode": 2,
                    "stories": [
                        {
                            "story_no": 1,
                            "nt_name": "溫情走心",
                            "places": [
                                {
                                    "place_id": "837a2e47-11ed-45ac-a6cc-85d9fedca5e0",
                                    "p_name": "臺中州廳",
                                    "is_hotel": 2,
                                    "is_hidden": 2,
                                    "type_list": [
                                        { "type_id": 6, "type_name": "文化問答型" }
                                    ]
                                },
                                {
                                    "place_id": "c23158b0-3b7c-4f5e-9283-e31f10798165",
                                    "p_name": "範例旅館",
                                    "is_hotel": 1,
                                    "is_hidden": 2,
                                    "type_list": [
                                        { "type_id": 8, "type_name": "e人訪談型" }
                                    ]
                                }
                            ]
                        },
                        {
                            "story_no": 2,
                            "nt_name": "幽默詼諧",
                            "places": [
                                {
                                    "place_id": "956be95e-d36f-4df8-b0cf-27a0012a600a",
                                    "p_name": "國立臺灣美術館",
                                    "is_hotel": 2,
                                    "is_hidden": 2,
                                    "type_list": [
                                        { "type_id": 3, "type_name": "創意攝影型" }
                                    ]
                                },
                                {
                                    "place_id": "374d051b-0a40-4219-8d6d-da5794164236",
                                    "p_name": "審計新村",
                                    "is_hotel": 2,
                                    "is_hidden": 2,
                                    "type_list": [
                                        { "type_id": 4, "type_name": "地方美食型" }
                                    ]
                                }
                            ]
                        },
                        {
                            "story_no": 3,
                            "nt_name": "懸疑推理",
                            "places": [
                                {
                                    "place_id": "223a3805-2127-401c-a789-a33284d3f8ce",
                                    "p_name": "刑務所演武場",
                                    "is_hotel": 2,
                                    "is_hidden": 2,
                                    "type_list": [
                                        { "type_id": 5, "type_name": "協作解謎型" }
                                    ]
                                },
                                {
                                    "place_id": "3f1f829a-f57d-4eb0-b054-143e45090444",
                                    "p_name": "忠信市場",
                                    "is_hotel": 2,
                                    "is_hidden": 1,
                                    "type_list": [
                                        { "type_id": 4, "type_name": "地方美食型" }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    }

class TaskOption(BaseModel):
    option_key: str
    option_context: str
    is_correct: int

class TaskClue(BaseModel):
    seat_no: int
    clue_text: str

# task_option / task_clue 沒設定時不輸出 (需搭配路由的 response_model_exclude_unset=True)
class GeneratedTask(BaseModel):
    task_type: int
    task_describe: str
    task_hint: str
    correct_answer: Optional[str] = None
    task_option: Optional[List[TaskOption]] = None
    task_clue: Optional[List[TaskClue]] = None

# 🆕 NPC 在節點親口說的台詞 (規格書以外的附加欄位)
class NodeDialogues(BaseModel):
    opening: str    # 玩家抵達時說的話
    success: str    # 玩家完成任務後說的話

class GeneratedNode(BaseModel):
    place_id: str
    sn_order: int
    sn_title: str
    location_codename: str
    sn_opening_text: str
    sn_success_text: str
    dialogues: Optional[NodeDialogues] = None   # 🆕
    tasks: List[GeneratedTask]

class StoryDetail(BaseModel):
    story_title: str
    story_prologue: str
    story_synopsis: str
    story_badge: List[str]

# 🆕 這份劇本的引導 NPC (規格書以外的附加欄位)；名稱避開上方對話節點用的 NpcInfo
class StoryNpc(BaseModel):
    name: str
    role: str
    intro: str

class GeneratedStoryItem(BaseModel):
    story_no: int
    story: StoryDetail
    npc: Optional[StoryNpc] = None              # 🆕
    nodes: List[GeneratedNode]

class StoryTaskResponse(BaseModel):
    stories: List[GeneratedStoryItem]

# ==========================================
# 商家 AI 顧問：輸入結構 (Request)
# ==========================================
class Neo4jAudienceData(BaseModel):
    cluster_name: str = Field(..., description="分群名稱，例如：年輕女性小資族")
    main_age: str = Field(..., description="主要年齡層，例如：29歲以下")
    female_pct: float = Field(..., description="女性比例 (0~1)")
    purpose_leisure_pct: float = Field(..., description="休閒目的比例 (0~1)")
    top_transports: str = Field(..., description="主要交通方式")
    mean_days: float = Field(..., description="平均停留天數")

class MerchantKpiData(BaseModel):
    task_completion_pr: int = Field(..., description="任務完成率 PR 值 (1~99)")
    social_share_pr: int = Field(..., description="Vlog社群分享率 PR 值 (1~99)")
    coupon_redemption_pr: int = Field(..., description="優惠兌換率 PR 值 (1~99)")

class MerchantAdvisorRequest(BaseModel):
    merchant_name: str = Field(..., description="商家名稱")
    category: str = Field(..., description="商家分類，例如：甜點冰品")
    operating_hours: str = Field(..., description="營業時間")
    current_task_type: str = Field(..., description="目前設定的解謎任務題型，例如：文化問答型")
    neo4j_audience: Neo4jAudienceData
    kpi: MerchantKpiData

    # 👇 加入這段，Swagger UI 就會自動帶入這個預設 JSON
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "merchant_name": "審計新村-甜點時光",
                    "category": "甜點冰品",
                    "operating_hours": "11:00-18:00",
                    "current_task_type": "文化問答型",
                    "neo4j_audience": {
                        "cluster_name": "公車散策年輕女性",
                        "main_age": "29歲以下",
                        "female_pct": 0.75,
                        "purpose_leisure_pct": 0.85,
                        "top_transports": "大眾運輸/步行",
                        "mean_days": 1.5
                    },
                    "kpi": {
                        "task_completion_pr": 85,
                        "social_share_pr": 35,
                        "coupon_redemption_pr": 60
                    }
                }
            ]
        }
    }

# ==========================================
# 商家 AI 顧問：輸出結構 (Response)
# ==========================================
class CompetitiveAnalysis(BaseModel):
    winning_at: str = Field(..., description="領先同業的指標與肯定")
    lagging_behind: str = Field(..., description="落後同業的指標與流失客源警告")

class MerchantAdvisorResponse(BaseModel):
    audience_profile: str = Field(..., description="一句話總結客群輪廓的白話文")
    competitive_analysis: CompetitiveAnalysis
    actionable_advice: List[str] = Field(..., description="營運、任務、社群三個面向的具體建議")