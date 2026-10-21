"""自然語言故事流程的業務例外。"""


class NoMatchingLocations(ValueError):
    """沒有可採用的候選，不代表程式發生未預期錯誤。"""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "no_matching_locations",
    ):
        super().__init__(message)
        self.reason_code = reason_code