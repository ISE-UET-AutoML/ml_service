import base64
from email.policy import HTTP
from pathlib import Path
from tkinter import Label
from turtle import title
from urllib import response
from fastapi import APIRouter, File, HTTPException, UploadFile
from httpx import head
import pandas
import requests
from sympy import content

import xmltodict

from settings.config import (
    label_studio_client,
    LABEL_STUDIO_HOST,
    LABEL_STUDIO_ACCESS_TOKEN,
)

from label_studio_sdk.label_interface import LabelInterface
from label_studio_sdk.label_interface.create import choices
from label_studio_sdk.projects import ProjectsCreateResponse
from label_studio_sdk.projects import ProjectsListResponse

from .LabelServiceRequest import ProjectCreateRequest, ProjectUploadCSVRequest

from .constants import PROJECT_TYPES, DataUploadType

from utils.save_file import save_file_to_public

router = APIRouter()


@router.get("/", response_model=ProjectsListResponse)
async def get_projects():
    return label_studio_client.projects.list()


@router.get("/{project_id}")
async def get_project(project_id: str):
    return label_studio_client.projects.get(project_id)


@router.post(
    "/create",
)
async def create_project(request: ProjectCreateRequest):

    label_interface = None
    match request.type:
        case PROJECT_TYPES.IMAGE_CLASSIFICATION:
            label_interface = LabelInterface.create(
                {
                    "image": "Image",
                    "label": choices(request.label_config.label_choices),
                }
            )
        case PROJECT_TYPES.TEXT_CLASSIFICATION:
            label_interface = LabelInterface.create(
                {
                    "text": "Text",
                    "label": choices(request.label_config.label_choices),
                }
            )
        case default:
            raise ValueError(
                f"Unknown/Currently not suported project type: {request.type}"
            )
    project = label_studio_client.projects.create(
        title=request.name, description=request.type, label_config=label_interface
    )

    return {"id": project.id}


@router.post("/{project_id}/delete")
def delete_project(project_id: str):
    return label_studio_client.projects.delete(project_id)


@router.post("/{project_id}/delete_tasks")
def delete_tasks(project_id: str, task_ids: list[str]):
    for task_id in task_ids:
        label_studio_client.tasks.delete(
            ids=task_id,
        )


@router.post("/{project_id}/delete_all_tasks")
def delete_all_tasks(project_id: str):
    response = requests.delete(
        f"{LABEL_STUDIO_HOST}/api/projects/{project_id}/tasks",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {LABEL_STUDIO_ACCESS_TOKEN}",
        },
    )
    return response.json()


@router.post("/{project_id}/upload/{upload_type}")
async def upload_project(
    project_id: str,
    upload_type: str,
    # request: ProjectUploadCSVRequest,
    files: list[UploadFile] = File(...),
):
    match upload_type:
        case DataUploadType.IMAGE_LABELED_FOLDER:
            return await upload_image_labeled_folder(project_id, files)

        case _:
            return await upload_image_labeled_folder(project_id, files)
            raise ValueError(
                f"Unknown/Currently not suported upload type: {upload_type}"
            )


@router.post("/{project_id}/upload_csv_files/")
# def upload_csv_files(
#     project_id: str, request: ProjectUploadCSVRequest, files: UploadFile = File(...)
# ):
#     pandas.read_csv(files.file)
#     if request.data_type == "Tabular":
#         raise HTTPException(status_code=500, detail="Tabular data type Not implemented")

#     pass


async def upload_image_labeled_folder(
    project_id: str, files: list[UploadFile] = File(...)
):
    print(project_id)
    import_tasks = []
    for file in files:
        # file_path = Path(file.filename)
        original_file_name = base64.b64decode(file.filename).decode("ascii")
        print(original_file_name)

        path = original_file_name.split("/")
        label, file_name = path[1], path[2]
        content = file.file.read()
        file_extension = original_file_name.split(".")[-1]
        file_route = save_file_to_public(content, file_extension)

        import_tasks.append({"image": file_route, "label": label})

    print(import_tasks)

    label_studio_client.projects.import_tasks(
        id=project_id,
        request=import_tasks,
        preannotated_from_fields=["label"],
    )

    await create_anotations_from_predictions(project_id, model_version="undefined")

    raise HTTPException(status_code=500, detail="Not implemented")
    return {"status": "success"}


from label_studio_sdk.data_manager import Filters, Column, Type, Operator


@router.get("/{project_id}/dataset")
async def export_task(project_id: str):

    tasks = label_studio_client.tasks.list(project=project_id, fields="all")
    # print(tasks)
    data = []
    for task in tasks.items:
        print(task.data)
        print(task.predictions)
        # You can access annotations in Label Studio JSON format
        print(task.annotations)
        data.append(
            {
                "data": task.data,
                "predictions": task.predictions,
                "annotations": task.annotations,
            }
        )
    return {"dataset": data}


@router.get("/{project_id}/simple_dataset")
async def export_task_simple(project_id: str, prediction_as_annotation: bool = False):
    """
    prediction_as_annotation: if annotation is not available, use prediction (if available) as annotation
    """
    project = label_studio_client.projects.get(project_id)
    label_config = xmltodict.parse(project.label_config)
    print(label_config)
    if "Choices" not in label_config["View"]:
        raise ValueError("Only choices type is supported")

    tasks = label_studio_client.tasks.list(project=project_id, fields="all")
    # print(tasks)
    data = []
    for task in tasks.items:
        anotation: list = task.annotations
        prediction: list = task.predictions

        if anotation.__len__() == 0 and prediction_as_annotation:
            anotation = prediction
        if anotation.__len__() == 0:
            continue  # skip task if no annotation or prediction

        anotation = anotation[0]["result"][0]

        if anotation["type"] != "choices":
            raise ValueError(f"Only choices type is supported, got {anotation['type']}")

        data.append(
            {
                "data": task.data,
                "anotation": anotation["value"]["choices"][0],
            }
        )
    return {
        "metadata": {
            "project": project_id,
            "label_config": label_config,
        },
        "dataset": data,
    }


async def create_anotations_from_predictions(
    project_id: str, model_version: str = None
):
    payload = {
        "filters": {"conjunction": "and", "items": []},
        "model_version": model_version,
        "ordering": [],
        "project": project_id,
        "selectedItems": {"all": True, "excluded": []},
    }
    response = requests.post(
        f"{LABEL_STUDIO_HOST}/api/dm/actions",
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Token {LABEL_STUDIO_ACCESS_TOKEN}",
        },
        params={"id": "predictions_to_annotations", "project": project_id},
        json=payload,
    )
    return response.json()


def update_project():
    pass
    # label_config = LabelInterface.create(
    #     {
    #         "image": "Image",
    #         "label": choices(["cat", "dog"]),
    #     }
    # )
    # label_studio_client.projects.update()
