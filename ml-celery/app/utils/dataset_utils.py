import os
from pathlib import Path
import splitfolders
import shutil
import glob
from typing import Union
import gdown
from zipfile import ZipFile


def split_data(
    input_folder: Path,
    output_folder,
    ratio=(0.8, 0.1, 0.1),
    seed=1337,
    group_prefix=None,
    move=False,
):
    """
    Splits a dataset into training, validation, and testing sets.

    Parameters:
    input_folder (str): Path to the dataset folder.
    output_folder (str): Path where the split available_checkpoint will be saved.
    ratio (tuple): A tuple representing the ratio to split (train, val, test).
    seed (int): Random seed for reproducibility.
    group_prefix (int or None): Prefix of group name to split files into different groups.
    move (bool): If True, move files instead of copying.

    Returns:
    None
    """
    try:
        splitfolders.ratio(
            input_folder,
            output=output_folder,
            seed=seed,
            ratio=ratio,
            group_prefix=group_prefix,
            move=move,
        )
        print("Data splitting completed successfully.")
    except Exception as e:
        print(f"An error occurred during data splitting: {e}")


def create_csv(directory: Path, output_file: Path):
    with open(output_file, mode="w") as f:
        f.write("image,label\n")
        for path, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    label = Path(path).name
                    f.write(f"{os.path.join(path, file)},{label}\n")


def remove_folders_except(user_dataset_path: Path, keep_folder: str):
    """
    Removes all subdirectories in the given directory except the specified folder.

    Args:
        user_dataset_path (Path): The path to the user's dataset directory.
        keep_folder (str): The name of the folder to keep.
    """
    for item in user_dataset_path.iterdir():
        if item.is_dir() and item.name != keep_folder:
            shutil.rmtree(item)
            print(f"Removed {item.name}")


def create_folder(user_dataset_path: Path):
    """
    Creates a folder in the user's dataset directory.

    Args:
        user_dataset_path (Path): The path to the user's dataset directory.
        folder_name (str): The name of the folder to create.
    """
    folder_path = user_dataset_path
    if not folder_path.exists():
        folder_path.mkdir()
        print(f"Created {user_dataset_path}")


def find_latest_model(user_model_path: str) -> Union[str, None]:
    """_summary_

    Args:
        user_model_path (str): _description_

    Returns:
        Union[str, None]: _description_
    """
    pattern = os.path.join(user_model_path, "**", "*.ckpt")
    list_of_files = glob.glob(pattern, recursive=True)
    return max(list_of_files, key=os.path.getctime) if list_of_files else None


def write_image_to_temp_file(image, temp_image_path):
    with open(temp_image_path, "wb") as buffer:
        buffer.write(image)


def model_size(user_model_path):
    pattern = os.path.join(user_model_path, "**", "*.ckpt")
    list_of_files = glob.glob(pattern, recursive=True)
    model_size = 0
    for file in list_of_files:
        model_size += os.path.getsize(file)
    return model_size


def download_dataset(dataset_dir: str, is_zip: bool, url: str, method: str):
    """
    Download dataset

    Args:
        dataset_dir: local folder where the dataset is going to be stored, should be **/dataset/
        is_zip: is object a folder? or a simple csv file?
        only tabular prediction task and text prediction task will have an data.csv file, other tasks will have multiple files
        url:
        method: where to download dataset from
    """
    os.makedirs(dataset_dir, exist_ok=True)
    datafile = ""
    if method == "gdrive":
        datafile = download_dataset_gdrive(dataset_dir, is_zip, url)

    if is_zip == False:
        return datafile
    else:
        with ZipFile(Path(datafile), "r") as zip_ref:
            zip_ref.extractall(dataset_dir)

            has_root_dir = True
            root_dir: str | None = None

            for subfile in zip_ref.namelist():
                path = subfile.split("/")

                if path[0].__contains__("."):
                    has_root_dir = False
                    break
                if root_dir is None:
                    root_dir = path[0]
                elif root_dir != path[0]:
                    has_root_dir = False
                    break
            if has_root_dir and root_dir is not None:
                return f"{dataset_dir}/{root_dir}"
            else:
                return dataset_dir


def download_dataset_gdrive(dataset_dir: str, is_zip: bool, url: str):
    dataset_url = f"https://drive.google.com/uc?id={url}"
    if is_zip:
        dataset_path = f"{dataset_dir}/data.zip"
        gdown.download(url=dataset_url, output=dataset_path, quiet=False)
        return dataset_path
    else:
        dataset_path = f"{dataset_dir}/data.csv"
        gdown.download(url=dataset_url, output=dataset_path, quiet=False)
        return dataset_path
