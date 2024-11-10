import os

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
from vastai import VastAI
from settings.config import VAST_AI_API_KEY

TEMP_DIR="./instance_ssh_keys"
vast_sdk = VastAI(api_key=VAST_AI_API_KEY)

def generate_ssh_key_pair(task_id):
    os.makedirs(TEMP_DIR, exist_ok=True)
    # Generate private key
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    # Serialize the private key
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption()
    )

    # Generate public key
    public_key = private_key.public_key()

    # Serialize the public key
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )

    # Save the keys to files named with the task_id
    private_key_filename = f"{TEMP_DIR}/{task_id}_id_rsa"
    public_key_filename = f"{TEMP_DIR}/{task_id}_id_rsa.pub"

    with open(private_key_filename, "wb") as f:
        f.write(private_key_pem)

    with open(public_key_filename, "wb") as f:
        f.write(public_key_pem)

    with open(public_key_filename, "rb") as f:
        public_key_string = f.read().decode("utf-8")

    print(f"SSH key pair generated and saved as '{private_key_filename}' (private key) and '{public_key_filename}' (public key).")

    return private_key_filename
    
# Example usage
if __name__ == "__main__":
    generate_ssh_key_pair("task123")

def attach_ssh_key_to_instance(task_id, instance_id):
    public_key_filename = f"{TEMP_DIR}/{task_id}_id_rsa.pub"
    with open(public_key_filename, "rb") as f:
        public_key_string = f.read().decode("utf-8")

    response = vast_sdk.attach_ssh(instance_id, public_key_string)
    
    return response