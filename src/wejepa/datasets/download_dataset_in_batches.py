""" code to download any dataset in batches """
from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as T
import torch.distributed as dist


import requests

url = "https://example.com/large_dataset.zip"  # Replace with your dataset URL
local_filename = "LAION-batch.zip"
chunk_size = 1024 * 1024  # 1 MB chunks

try:
    with requests.get(url, stream=True) as r:
        r.raise_for_status()  # Raise an exception for bad status codes
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
    print(f"Dataset downloaded in chunks to {local_filename}")
except requests.exceptions.RequestException as e:
    print(f"Error downloading dataset: {e}")

def main() -> None:
   



if __name__ == "__main__":
    main()