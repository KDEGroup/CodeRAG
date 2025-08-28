from pydantic import BaseModel
class SuperviseTrainingDataItem(BaseModel):
    query: str
    candidates: list[str]
    option: int
    messages: list[dict]
    expected: str


class SuperviseTrainingData(BaseModel):
    data_list: list[SuperviseTrainingDataItem]
