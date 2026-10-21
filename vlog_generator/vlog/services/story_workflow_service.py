"""自然語言故事的結果分流。

保留既有生成器。
尚未接入已驗證劇本庫，因此不回傳假備援劇本。
"""

import logging


from api.story_generation_schemas import (
    StoryGenerationResult,
)
from services.story_workflow_errors import (
    NeedsTravelClarification,
    NoMatchingLocations,
    StoryGenerationExhausted,
)


logger = logging.getLogger(__name__)


def _result(
    outcome,
    message,
    reason_code=None,
    story_payload=None,
    alternatives=None,
):
    validated = StoryGenerationResult.model_validate(
        {
            "outcome": outcome,
            "message": message,
            "reason_code": reason_code,
            "story_payload": story_payload,
            "alternatives": alternatives or [],
        }
    )

    return validated.model_dump(
        mode="json",
        exclude_unset=True,
    )


def _guidance_options():
    return [
        {
            "action": "submit_prompt",
            "label": "探索自然景觀",
            "user_prompt": "我們想探索自然景觀",
        },
        {
            "action": "submit_prompt",
            "label": "探索地方文史",
            "user_prompt": "我們想探索地方文史",
        },
        {
            "action": "edit_request",
            "label": "補充地區或想體驗的內容",
        },
    ]


def _early_guidance(user_prompt):
    """處理明確預設值與簡單模糊輸入，不是完整意圖辨識。"""
    normalized = user_prompt.casefold().strip()

    placeholders = {
        "",
        "string",
        "null",
        "none",
        "undefined",
        "test",
        "測試",
    }

    vague_inputs = {
        "隨便",
        "不知道",
        "都可以",
        "隨便安排",
        "幫我安排",
        "不知道去哪",
        "不知道去哪裡",
    }

    if normalized in placeholders:
        return _result(
            outcome="needs_clarification",
            message=(
                "請輸入想去的地區或想體驗的內容，"
                "也可以先選一個探索方向。"
            ),
            reason_code="invalid_or_placeholder_input",
            alternatives=_guidance_options(),
        )

    if normalized in vague_inputs:
        return _result(
            outcome="needs_clarification",
            message=(
                "還沒有明確的旅遊方向，"
                "你可以先選主題，或補充想去的地區。"
            ),
            reason_code="needs_travel_preferences",
            alternatives=_guidance_options(),
        )

    return None


def internal_error_result():
    """對外安全訊息；未預期錯誤由路由記錄完整日誌。"""
    return _result(
        outcome="temporarily_unavailable",
        message=(
            "這次處理遇到未預期問題，"
            "尚未提供可用劇本，請稍後重試。"
        ),
        reason_code="internal_error",
        alternatives=[
            {
                "action": "retry",
                "label": "稍後重試原需求",
            },
            {
                "action": "edit_request",
                "label": "修改需求",
            },
        ],
    )


def generate_story_workflow(user_prompt: str) -> dict:
    user_prompt = str(user_prompt or "").strip()

    guidance = _early_guidance(user_prompt)

    if guidance is not None:
        return guidance

    # 有具體需求，才載入生成器與資料庫例外。
    from services.blueprint_auto_service import (
        generate_story_from_text,
    )
    from neo4j.exceptions import (
        ServiceUnavailable,
        SessionExpired,
    )

    try:
        story = generate_story_from_text(
            user_prompt
        )

    except NeedsTravelClarification as exc:
        logger.info(
            "旅遊需求需要補充：%s",
            exc,
        )

        return _result(
            outcome="needs_clarification",
            message=(
                "目前還無法確認你想去哪裡或體驗什麼。"
                "請補充一個地區或活動，"
                "例如「臺中散步」或「兩個人想看雲海」。"
            ),
            reason_code="needs_travel_preferences",
            alternatives=_guidance_options(),
        )

    except NoMatchingLocations as exc:
        logger.info(
            "故事候選不足：%s",
            exc,
        )

        alternatives = [
            {
                "action": "edit_request",
                "label": "補充縣市或更具體的景觀類型",
            },
            {
                "action": "submit_prompt",
                "label": "改找自然景觀",
                "user_prompt": "我們想探索自然景觀",
            },
            {
                "action": "submit_prompt",
                "label": "改找地方文史",
                "user_prompt": "我們想探索地方文史",
            },
        ]

        current_prompt = user_prompt.strip().casefold()

        alternatives = [
            option
            for option in alternatives
            if not (
                option["action"] == "submit_prompt"
                and option["user_prompt"].strip().casefold()
                == current_prompt
            )
        ]

        return _result(
            outcome="no_match",
            message=(
                "本次候選未能確認符合需求。"
                "這不代表資料庫或臺灣沒有相關地點；"
                "你可以補充縣市、改成更具體的景觀類型，"
                "或選擇其他探索方向。"
            ),
            reason_code="no_supported_candidate",
            alternatives=alternatives,
        )

    except StoryGenerationExhausted as exc:
        logger.warning(
            "故事生成重試耗盡：%s",
            exc,
        )

        return _result(
            outcome="temporarily_unavailable",
            message=(
                "客製化劇本未通過品質檢查，"
                "目前尚未提供相容的已驗證備援劇本。"
                "你可以修改需求或稍後重試。"
            ),
            reason_code="generation_validation_exhausted",
            alternatives=[
                {
                    "action": "retry",
                    "label": "稍後重試原需求",
                },
                {
                    "action": "edit_request",
                    "label": "修改需求",
                },
            ],
        )

    except (ServiceUnavailable, SessionExpired) as exc:
        logger.warning(
            "地點資料服務暫時不可用：%s",
            exc,
        )

        return _result(
            outcome="temporarily_unavailable",
            message=(
                "地點資料服務暫時無法使用，"
                "目前無法可靠產生實地劇本，"
                "請稍後重試。"
            ),
            reason_code="location_service_unavailable",
            alternatives=[
                {
                    "action": "retry",
                    "label": "稍後重試原需求",
                }
            ],
        )

    # 完整劇本仍由 StoryGenerationResult 中的
    # PublicStoryTaskResponse 驗證，不接受空劇本。
    return _result(
        outcome="generated",
        message="已完成客製化劇本。",
        story_payload=story,
    )