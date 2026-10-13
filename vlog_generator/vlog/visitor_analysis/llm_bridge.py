from __future__ import annotations

import logging
from typing import Any, Optional

from visitor_analysis.settings import get_settings

logger = logging.getLogger(__name__)


class LlmBridgeError(Exception):
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class VlogLlmBridge:
    """呼叫 vlog 地端 Llama（services.llm_service.generate_chat_text）。"""

    def generate(
        self,
        *,
        system_prompt: Optional[str] = None,
        user_prompt: str,
        temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        user = (user_prompt or "").strip()
        if not user:
            raise LlmBridgeError("invalid_request", "user_prompt 不可為空白。")

        settings = get_settings()
        temp = (
            float(settings.strategy_temperature)
            if temperature is None
            else float(temperature)
        )
        tokens = (
            int(settings.strategy_max_new_tokens)
            if max_new_tokens is None
            else int(max_new_tokens)
        )

        try:
            from services.llm_service import generate_chat_text
        except Exception as exc:
            raise LlmBridgeError("import_failed", f"無法載入 vlog LLM：{exc}") from exc

        try:
            raw_text = generate_chat_text(
                system_prompt=system_prompt,
                user_prompt=user,
                temperature=temp,
                max_new_tokens=tokens,
            )
        except Exception as exc:
            logger.exception("vlog LLM 生成失敗")
            raise LlmBridgeError("generation_failed", f"LLM 生成失敗：{exc}") from exc

        raw_text = (raw_text or "").strip()
        if not raw_text:
            raise LlmBridgeError("empty_output", "LLM 回傳空白內容。")

        return {
            "status": "ready",
            "message": "生成完成",
            "content": raw_text,
            "raw_text": raw_text,
        }
