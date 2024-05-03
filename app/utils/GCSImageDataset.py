import io

from zipfile import ZipFile
from PIL import Image
from google.cloud import storage
from torch.utils.data import Dataset

from app.config import config


# to use Webdataset, Optimize string big data
# Not done yet @DuongNam
class GCSImageDataset(Dataset):
    def __init__(self, gcs_paths, transform=None):
        """
        Args:
            gcs_paths (list): List of GCS URLs for the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.gcs_paths = gcs_paths
        self.transform = transform
        self.storage_client = storage.Client().from_service_account_json(config.gcp_storage)

    def __len__(self):
        return len(self.gcs_paths)

    def __getitem__(self, idx):
        gcs_path = self.gcs_paths[idx]
        bucket_name, blob_name = self.parse_gcs_url(gcs_path)

        bucket = self.storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        image_bytes = blob.download_as_bytes()

        # Convert bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image

    @staticmethod
    def parse_gcs_url(gcs_url):
        """Parse GCS URL into bucket and blob names."""
        if gcs_url.startswith('gs://'):
            _, bucket_name, blob_name = gcs_url.split('/', 3)
            return bucket_name, blob_name
        raise ValueError(f'Invalid GCS URL: {gcs_url}')
