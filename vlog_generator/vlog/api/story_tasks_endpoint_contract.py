"""僅供 /api/stories/tasks 使用的精簡輸出契約。

共用生成器仍回傳完整劇本。
本模組建立獨立輸出，不修改原始結果，
不影響自動故事 v2 的 npc 與 dialogues。
"""

import copy

from pydantic import BaseModel, ConfigDict, Field

from api.story_task_schemas import (
    PublicGeneratedTask,
    PublicStoryDetail,
    PublicStoryTaskRequest,
    PublicStoryTaskResponse,
)


class _EndpointStrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class StoryTasksEndpointNode(_EndpointStrictModel):
    place_id: str = Field(min_length=1)
    sn_order: int = Field(ge=1)
    sn_title: str = Field(min_length=1)
    location_codename: str = Field(min_length=1)
    sn_opening_text: str = Field(min_length=1)
    sn_success_text: str = Field(min_length=1)
    tasks: list[PublicGeneratedTask] = Field(min_length=1)


class StoryTasksEndpointResponse(_EndpointStrictModel):
    story: PublicStoryDetail
    nodes: list[StoryTasksEndpointNode] = Field(min_length=1)


def build_story_tasks_endpoint_result(
    generated: dict,
    request_data: dict,
) -> dict:
    """將完整劇本轉成此端點專用的 story + nodes。

    不變更共用模型、不原地刪除原始資料。
    僅進行結構及輸入輸出對應驗證，
    不宣稱已驗證問答事實或協作謎題可解性。
    """
    request = PublicStoryTaskRequest.model_validate(
        request_data
    )

    full_response = PublicStoryTaskResponse.model_validate(
        copy.deepcopy(generated)
    )

    if len(full_response.nodes) != len(request.places):
        raise ValueError(
            "輸出的節點數量與輸入景點數量不一致"
        )

    projected_nodes = []

    for node_index, (place, node) in enumerate(
        zip(request.places, full_response.nodes),
        start=1,
    ):
        if node.place_id != place.place_id:
            raise ValueError(
                f"第 {node_index} 站的景點識別碼"
                "或順序與輸入不一致"
            )

        if node.sn_order != node_index:
            raise ValueError(
                f"第 {node_index} 站的 sn_order 不正確"
            )

        expected_types = [
            item.type_id
            for item in place.type_list
        ]

        actual_types = [
            task.task_type
            for task in node.tasks
        ]

        if actual_types != expected_types:
            raise ValueError(
                f"第 {node_index} 站的任務數量、"
                "類型或順序與輸入不一致"
            )

        projected_tasks = []

        for task_index, task in enumerate(
            node.tasks,
            start=1,
        ):
            task_data = task.model_dump(
                mode="json",
                exclude_unset=True,
            )

            # 此端點要求所有任務明確包含 correct_answer。
            task_data["correct_answer"] = (
                task.correct_answer
            )

            if task.task_type == 5:
                clues = task.task_clue or []

                expected_seats = list(
                    range(1, request.party_size + 1)
                )

                actual_seats = [
                    clue.seat_no
                    for clue in clues
                ]

                if actual_seats != expected_seats:
                    raise ValueError(
                        f"第 {node_index} 站"
                        f"第 {task_index} 個任務的"
                        "線索數量或座位編號不符合玩家人數"
                    )

            projected_tasks.append(task_data)

        projected_nodes.append({
            "place_id": node.place_id,
            "sn_order": node.sn_order,
            "sn_title": node.sn_title,
            "location_codename": node.location_codename,
            "sn_opening_text": node.sn_opening_text,
            "sn_success_text": node.sn_success_text,
            "tasks": projected_tasks,
        })

    payload = {
        "story": full_response.story.model_dump(
            mode="json"
        ),
        "nodes": projected_nodes,
    }

    response = StoryTasksEndpointResponse.model_validate(
        payload
    )

    return response.model_dump(
        mode="json",
        exclude_unset=True,
    )