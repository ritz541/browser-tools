from pydantic import BaseModel, Field

from app.config import DEFAULT_MODEL


class StartSessionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)


class PromptRequest(BaseModel):
    text: str


class NavigateRequest(BaseModel):
    url: str


class ClickRequest(BaseModel):
    selector: str | None = None
    text: str | None = None
    nth: int = 0


class TypeRequest(BaseModel):
    selector: str
    text: str
    clear: bool = True
    press_enter: bool = False


class FormField(BaseModel):
    selector: str
    value: str


class FillFormRequest(BaseModel):
    fields: list[FormField]
    submit_selector: str | None = None


class SubmitRequest(BaseModel):
    selector: str | None = None


class ScrollRequest(BaseModel):
    direction: str = "down"
    amount: int = 500
