"""/api/stories/tasks 專用：公開回應包含 story、npc 與 nodes。"""

from typing import Literal, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from api.schemas import SingleStoryTaskRequest


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PublicStoryTaskRequest(SingleStoryTaskRequest):
    @model_validator(mode="after")
    def validate_task_plan(self):
        ids = [place.place_id for place in self.places]

        if len(ids) != len(set(ids)):
            raise ValueError("景點識別碼不可重複")

        for place in self.places:
            if (
                place.is_hotel not in (1, 2)
                or place.is_hidden not in (1, 2)
            ):
                raise ValueError(
                    "is_hotel 與 is_hidden 只能是 1 或 2"
                )

            if not place.type_list:
                raise ValueError("每個景點至少指定一種任務類型")

            task_types = [
                item.type_id
                for item in place.type_list
            ]

            if any(
                item not in range(1, 9)
                for item in task_types
            ):
                raise ValueError("任務類型只能介於 1 與 8")

            if len(task_types) != len(set(task_types)):
                raise ValueError("同一景點的任務類型不可重複")

            if 5 in task_types and self.party_size < 2:
                raise ValueError("協作解謎至少需要兩位玩家")

        return self


class PublicStoryNpc(_StrictModel):
    name: str = Field(min_length=1)
    role: str = Field(min_length=1)
    intro: str = Field(min_length=1)

    @field_validator("name", "role", "intro")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("角色欄位不可只有空白")

        return value


class PublicNodeDialogues(_StrictModel):
    opening: str = Field(min_length=1)
    success: str = Field(min_length=1)

    @field_validator("opening", "success")
    @classmethod
    def validate_nonblank_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError("角色台詞不可只有空白")

        return value


class PublicStoryDetail(_StrictModel):
    story_title: str = Field(min_length=1)
    story_prologue: str = Field(min_length=1)
    story_synopsis: str = Field(min_length=1)
    story_badge: list[str] = Field(min_length=1)


class PublicTaskOption(_StrictModel):
    option_key: Literal["A", "B", "C", "D"]
    option_context: str = Field(min_length=1)
    is_correct: Literal[0, 1]


class PublicTaskClue(_StrictModel):
    seat_no: int = Field(ge=1)
    clue_text: str = Field(min_length=1)


class PublicGeneratedTask(_StrictModel):
    task_type: Literal[1, 2, 3, 4, 5, 6, 7, 8]
    task_describe: str = Field(min_length=1)
    task_hint: str = Field(min_length=1)
    correct_answer: Optional[str] = None
    task_option: Optional[list[PublicTaskOption]] = None
    task_clue: Optional[list[PublicTaskClue]] = None

    @model_validator(mode="after")
    def validate_task_shape(self):
        if (
            not self.task_describe.strip()
            or not self.task_hint.strip()
        ):
            raise ValueError("任務與提示不可只有空白")

        if self.task_type in (6, 7):
            if self.correct_answer is not None:
                raise ValueError(
                    "選擇題正解由 is_correct 表示"
                )

            if not self.task_describe.rstrip().endswith(
                ("？", "?")
            ):
                raise ValueError("選擇題必須是問句")

            options = self.task_option or []

            if len(options) != 4:
                raise ValueError("選擇題必須恰有四個選項")

            if [
                option.option_key
                for option in options
            ] != list("ABCD"):
                raise ValueError(
                    "選項代碼必須依序為 A、B、C、D"
                )

            if len({
                option.option_context.strip()
                for option in options
            }) != 4:
                raise ValueError("選項內容不可重複")

            if sum(
                option.is_correct
                for option in options
            ) != 1:
                raise ValueError("選擇題必須恰有一個正解")

            if "task_clue" in self.model_fields_set:
                raise ValueError("選擇題不輸出 task_clue")

        elif self.task_type == 5:
            if (
                not self.correct_answer
                or not self.correct_answer.strip()
            ):
                raise ValueError("協作解謎必須提供答案")

            if not self.task_clue:
                raise ValueError("協作解謎必須提供線索")

            if "task_option" in self.model_fields_set:
                raise ValueError(
                    "協作解謎不輸出 task_option"
                )

        else:
            if self.correct_answer is not None:
                raise ValueError(
                    "此類型的 correct_answer 必須是 null"
                )

            if {
                "task_option",
                "task_clue",
            } & self.model_fields_set:
                raise ValueError(
                    "此類型不輸出選項或協作線索"
                )

        return self


class PublicGeneratedNode(_StrictModel):
    place_id: str = Field(min_length=1)
    sn_order: int = Field(ge=1)
    sn_title: str = Field(min_length=1)
    location_codename: str = Field(min_length=1)
    sn_opening_text: str = Field(min_length=1)
    sn_success_text: str = Field(min_length=1)
    dialogues: PublicNodeDialogues
    tasks: list[PublicGeneratedTask] = Field(min_length=1)


class PublicStoryTaskResponse(_StrictModel):
    story: PublicStoryDetail
    npc: PublicStoryNpc
    nodes: list[PublicGeneratedNode] = Field(min_length=1)