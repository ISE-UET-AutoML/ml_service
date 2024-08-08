from typing import Optional
from pydantic import BaseModel, Field

LABEL_TYPES = ["multi_class", "obj_detection", "img_segmentation", "regression"]


class LabelConfig(BaseModel):
    # properties: dict = Field(
    #     {
    #         "_value": [],
    #         "_class": [],
    #         "_image": [],
    #         "_text": [],
    #         "_audio": [],
    #         "_video": [],
    #         "_ignore": [],
    #         "label": "label",
    #     },
    #     title="Properties / columns names",
    # )
    label_type: str = Field(LABEL_TYPES[0], title="Label type")
    label_choices: list[str] = Field(
        [], title="Label choices", description="Label choices, for classification tasks"
    )


class ProjectCreateRequest(BaseModel):
    name: str = Field(..., title="Project name", description="Project name")
    type: str = Field(..., title="Project type", description="Project type")
    label_config: LabelConfig = Field(...)
