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


class ProjectUploadCSVRequest(BaseModel):
    # data_type: Image, Text, Audio, Video, Tabular
    data_type: str = Field(default="Image", title="Data type", description="Data type")
    data_column_args: str | dict = Field(
        default={
            "image": "Image",
            "text": "Text",
        },
        title="Data arguments",
        description=(
            "if a single column is used, pass the column name as a string, "
            "if multiple columns are used, "
            "pass a dictionary with column names as keys, "
            "the values will be used as configuraion for the column",
        ),
    )
    label_type: str | None = Field(
        default="Choices", title="Label", description="Label"
    )
    # if label_type is None, then the label column will be ignored
    label_column_name: str = Field(default="label")
    label_choises: list[str] = Field(
        ..., title="Label choices", description="Label choices"
    )
