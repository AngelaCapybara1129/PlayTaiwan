import asyncio
import logging
import time
import uuid
from typing import Awaitable, Callable, Optional

from fastapi import HTTPException


logger = logging.getLogger(__name__)

JobFactory = Callable[[], Awaitable[dict]]


class JobQueue:
    def __init__(
        self,
        max_pending: int = 20,
        result_ttl: int = 3600,
    ):
        self.max_pending = max_pending
        self.result_ttl = result_ttl

        self._queue = None
        self._worker = None
        self._jobs = {}
        self._accepting = False

    async def start(self):
        if self._worker is not None:
            return

        self._queue = asyncio.Queue(
            maxsize=self.max_pending
        )
        self._accepting = True
        self._worker = asyncio.create_task(
            self._run(),
            name="ai-job-worker",
        )

        print(
            f"✅ AI 佇列已啟動："
            f"同時執行 1 個，未完成任務上限 {self.max_pending}"
        )

    async def stop(self):
        self._accepting = False

        if self._worker is None:
            return

        # 正常關閉時，等待已接受的工作完成。
        # 不以 cancel/to_thread 逾時假裝底層工作已停止。
        print("⏳ 等待已接受的 AI 任務完成後關閉...")
        await self._queue.join()

        self._queue.put_nowait(None)
        await self._worker
        self._worker = None

    def _prune(self):
        now = time.time()

        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if (
                job["status"] in ("done", "error")
                and now - job["updated_at"] > self.result_ttl
            )
        ]

        for job_id in expired:
            del self._jobs[job_id]

    def submit(
        self,
        task_type: str,
        factory: JobFactory,
        task_id: Optional[str] = None,
    ):
        if not self._accepting:
            raise HTTPException(
                status_code=503,
                detail="任務服務尚未就緒或正在關閉",
                headers={"Retry-After": "10"},
            )

        self._prune()

        pending = sum(
            job["status"] in ("queued", "running")
            for job in self._jobs.values()
        )

        if pending >= self.max_pending:
            raise HTTPException(
                status_code=503,
                detail="目前任務已滿，請稍後重試",
                headers={"Retry-After": "15"},
            )

        task_id = task_id or uuid.uuid4().hex

        if task_id in self._jobs:
            raise HTTPException(
                status_code=409,
                detail="任務 ID 已存在",
            )

        now = time.time()

        self._jobs[task_id] = {
            "task_id": task_id,
            "task_type": task_type,
            "status": "queued",
            "stage": "等待處理",
            "created_at": now,
            "updated_at": now,
            "result": None,
            "error": None,
        }

        try:
            # submit 不包含 await，檢查容量與入列不會在此中途讓出控制權。
            self._queue.put_nowait((task_id, factory))
        except asyncio.QueueFull:
            del self._jobs[task_id]
            raise HTTPException(
                status_code=503,
                detail="目前任務已滿，請稍後重試",
                headers={"Retry-After": "15"},
            )

        return self.get(task_id)

    def stage(self, task_id: str, text: str):
        job = self._jobs[task_id]
        job["stage"] = text
        job["updated_at"] = time.time()

    def get(self, task_id: str):
        self._prune()

        job = self._jobs.get(task_id)

        if job is None:
            raise HTTPException(
                status_code=404,
                detail="找不到任務，可能已過期或服務已重啟",
            )

        result = dict(job)
        result["check_url"] = f"/api/jobs/{task_id}"

        if job["status"] == "queued":
            queued_ids = [
                key
                for key, item in self._jobs.items()
                if item["status"] == "queued"
            ]
            result["queue_position"] = queued_ids.index(task_id) + 1
        else:
            result["queue_position"] = None

        return result

    async def _run(self):
        while True:
            item = await self._queue.get()

            try:
                if item is None:
                    return

                task_id, factory = item
                job = self._jobs[task_id]

                job.update(
                    status="running",
                    stage="開始處理",
                    updated_at=time.time(),
                )

                try:
                    result = await factory()

                except Exception:
                    logger.exception(
                        "AI 任務失敗：%s", task_id
                    )

                    job.update(
                        status="error",
                        stage="處理失敗",
                        error=(
                            "任務生成失敗，請聯絡管理員並提供任務 ID"
                        ),
                        updated_at=time.time(),
                    )

                else:
                    job.update(
                        status="done",
                        stage="完成",
                        result=result,
                        updated_at=time.time(),
                    )

            finally:
                self._queue.task_done()


job_queue = JobQueue(
    max_pending=20,
    result_ttl=3600,
)