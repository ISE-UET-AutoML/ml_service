import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import requests

IMAGE_SIZE = 224


# Apply softmax to each row (each sample's logits)
def softmax(logits):
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def preprocess_image(data):
    # Preprocessing function
    transform = transforms.Compose([
        transforms.Resize(size=(IMAGE_SIZE, IMAGE_SIZE)),  # Adjust size as needed
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    def preprocess_image(image_input):
        if isinstance(image_input, str):
            try:
                img = Image.open(requests.get(image_input, stream=True).raw)
            except Exception as e:
                return None
        else:
            img = Image.fromarray(image_input, 'RGB')
        img = transform(img)
        return img


    images = [preprocess_image(image_input) for image_input in data]
    image_tensor = np.stack(images)
    valid_nums = np.array([1] * len(images), dtype=np.int64) # Create the timm_image_image_valid_num tensor
    
    image_tensor = np.expand_dims(image_tensor, axis=1)
    return image_tensor, valid_nums


