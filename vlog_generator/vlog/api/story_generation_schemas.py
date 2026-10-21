"""自然語言劇本生成的業務結果。"""

from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)

from api.story_task_schemas import PublicStoryTaskResponse


class StoryAlternative(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Literal[
        "submit_prompt",
        "edit_request",
        "retry",
    ]

    label: str = Field(min_length=1)
    user_prompt: Optional[str] = None

    @model_validator(mode="after")
    def validate_action(self):
        self.label = self.label.strip()

        if not self.label:
            raise ValueError("選項文字不可只有空白")

        if self.action == "submit_prompt":
            if (
                not self.user_prompt
                or not self.user_prompt.strip()
            ):
                raise ValueError(
                    "submit_prompt 必須提供 user_prompt"
                )

            self.user_prompt = self.user_prompt.strip()

        elif self.user_prompt is not None:
            raise ValueError(
                "此操作不應附帶 user_prompt"
            )

        return self


class StoryGenerationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    outcome: Literal[
        "generated",
        "needs_clarification",
        "no_match",
        "fallback",
        "temporarily_unavailable",
    ]

    message: str = Field(min_length=1)
    reason_code: Optional[str] = None
    story_payload: Optional[PublicStoryTaskResponse] = None

    alternatives: list[StoryAlternative] = Field(
        default_factory=list
    )

    @model_validator(mode="after")
    def validate_result_shape(self):
        self.message = self.message.strip()

        if not self.message:
            raise ValueError("結果訊息不可只有空白")

        if self.reason_code is not None:
            self.reason_code = self.reason_code.strip()

        needs_story = self.outcome in {
            "generated",
            "fallback",
        }

        if needs_story and self.story_payload is None:
            raise ValueError(
                "generated 或 fallback 必須包含完整劇本"
            )

        if not needs_story and self.story_payload is not None:
            raise ValueError(
                "引導、無結果或服務異常不可附帶劇本"
            )

        if (
            self.outcome != "generated"
            and not self.reason_code
        ):
            raise ValueError(
                "非 generated 結果必須提供 reason_code"
            )

        return self