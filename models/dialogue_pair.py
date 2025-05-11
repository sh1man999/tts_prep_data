from pydantic import BaseModel, Field


class DialoguePair(BaseModel):
    id: int = Field(..., description="Уникальный идентификатор пары")
    user_query: str = Field(..., description="Запрос пользователя")
    ai_response: str = Field(..., description="Ответ ИИ на запрос")

    def to_jsonl(self):
        return self.model_dump_json()+'\n'
