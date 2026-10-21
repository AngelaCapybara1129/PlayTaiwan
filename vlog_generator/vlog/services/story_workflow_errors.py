"""故事工作流程的可辨識例外。"""


class NeedsTravelClarification(ValueError):
    """目前無法取得足夠的旅遊條件，需要使用者補充。"""


class NoMatchingLocations(ValueError):
    """已有可搜尋需求，但沒有符合條件的可用候選。"""


class StoryGenerationExhausted(RuntimeError):
    """模型生成重試耗盡，未取得合格劇本。"""