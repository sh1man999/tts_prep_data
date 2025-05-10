from pydantic import BaseModel


class NeuralProcessedText(BaseModel):
    original_text: str
    processed_text: str
    quality_score: float
    summary: str

    def to_jsonl(self):
        return self.model_dump_json() + '\n'
