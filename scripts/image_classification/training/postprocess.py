import requests
import json
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import os


def upload_file_to_s3(model_path, presign_url_config):
    try:
        # Open the file for uploading
        with open(model_path, 'rb') as f:
            files = {'file': (model_path, f)}
            http_response = requests.post(presign_url_config['url'], data=presign_url_config['fields'], files=files)
            
            # Check for HTTP errors
            if http_response.status_code != 204:
                print(f"Failed to upload file. HTTP Status: {http_response.status_code}")
                print(f"Response content: {http_response.text}")
                return {"error": f"Failed to upload file. HTTP Status: {http_response.status_code}"}
            
            print("File successfully uploaded to S3.")
            return {"success": True}

    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the file upload: {e}")
        return {"error": str(e)}

    except FileNotFoundError:
        print(f"File not found: {model_path}")
        return {"error": "File not found"}

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": str(e)}


def get_fit_history(model_path):
    event_file = ""
    model_name = model_path.split("/")[-1]
    model_path = model_path.replace(model_name, "model")
    print(model_path)
    for file in os.listdir(model_path):
        if file.startswith("events.out.tfevents"):
            event_file = f"{model_path}/{file}"
            break
    if event_file == "":
        return {"error": "No event file found"}

    print(event_file)
    ea = EventAccumulator(event_file).Reload()

    tags = ea.Tags()
    scalars = tags["scalars"]
    scalars_data = {}
    for scalar in scalars:
        df = pd.DataFrame(ea.Scalars(scalar), index=None)
        csv = df.to_csv()
        scalars_data[scalar] = csv

    fit_history_json = {"fit_history": {"scalars": scalars_data}}
    fit_history_path = model_path.replace("model", f"{model_name.replace('.zip', '')}_fit_history.json")
    print(fit_history_json)
    with open(fit_history_path, 'w') as f:
        json.dump(fit_history_json, f, ensure_ascii=False, indent=4)
    return fit_history_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, default="./trained_model.zip")
    parser.add_argument("--saved-model-url", type=str, required=True)
    parser.add_argument("--fit-history-url", type=str, required=True)

    args = parser.parse_args()
    saved_model_url = json.loads(args.saved_model_url)
    fit_history_url = json.loads(args.fit_history_url)

    fit_history_path = get_fit_history(args.model_path)

    # Upload the model to S3 and handle errors
    upload_response = upload_file_to_s3(args.model_path, saved_model_url)
    if "error" in upload_response:
        print(f"Error during model upload: {upload_response['error']}")
    else:
        print(f"Model successfully uploaded: {args.model_path}")

    # Upload the fit history to S3 and handle errors
    upload_response = upload_file_to_s3(fit_history_path, fit_history_url)

    if "error" in upload_response:
        print(f"Error during fit history upload: {upload_response['error']}")
    else:
        print(f"Fit history successfully uploaded: {fit_history_path}")
    
    print("Postprocess done!")
