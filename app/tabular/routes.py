from fastapi import FastAPI, HTTPException, APIRouter
import os
from jinja2 import Template

router = APIRouter()


def get_template_file(path):
    """
    Read the content of a file and return it as a string
    """
    with open(path, 'r', encoding='UTF-8') as file:
        return file.read()


def save_file(content, dst_path: str):
    """
    Save the content to a file to the specified path
    """
    with open(dst_path, 'w', encoding='UTF-8') as file:
        file.write(content)
    return dst_path


# provision EC2 g5.2x large instance and download data from s3
@router.post("/api/tabular_trainer/resource_provision", tags=["tabular_trainer"])
def handler(bucket_name: str = "lexuanan18102004", project_name: str = "titanic-survive"):
    try:
        import time
        start = time.time()
        # TODO : Add task to this (Currently default to tabular classification task)
        data = {
            "bucket_name": bucket_name,
            "project_name": project_name
        }

        user_terraform_dir = f"D:/tmp/{bucket_name}/{project_name}"
        os.makedirs(user_terraform_dir, exist_ok=True) if not os.path.exists(user_terraform_dir) else None

        absolute_path = os.path.abspath(os.path.dirname(__file__))

        ec2_template = get_template_file(
            f"{absolute_path}/templates/ec2.template.tf")

        jinja2_template = Template(ec2_template)

        content = jinja2_template.render(data)
        save_file(content, f"{user_terraform_dir}/main.tf")

        script_template = get_template_file(
            f"{absolute_path}/templates/user_data.sh.tpl")

        jinja2_template = Template(script_template)

        content = jinja2_template.render(data)

        save_file(content, f"{user_terraform_dir}/user_data.sh.tpl")

        # TODO: bug here can not use the cached plugin
        os.system(f"cd {user_terraform_dir} && "
                  f"export TF_PLUGIN_CACHE_DIR=\"$HOME/.terraform.d/plugin-cache\" && "
                  f"terraform init && terraform apply -auto-approve")

    except Exception as e:
        return HTTPException(status_code=500, detail=str(e))

    return {
        "message": f"Resource provisioned successfully with bucket name {bucket_name} and project name {project_name} in {time.time() - start} seconds"
    }
