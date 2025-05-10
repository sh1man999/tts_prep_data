from pydantic import BaseModel


class BaseRow(BaseModel):
    id: str
    text: str

    def to_jsonl(self):
        return self.model_dump_json()+'\n'

class HfRow(BaseRow):
    source: str
    file_name: str
    style: str = "default"
    voice: str