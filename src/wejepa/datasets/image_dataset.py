from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """Simple dataset for loading images from a directory and list."""
    def __init__(self, image_dir, image_list, labels=None, resolution=224):
        self.image_dir = Path(image_dir)
        self.image_list = image_list
        self.labels = labels
        self.resolution = resolution
    def __len__(self):
        return len(self.image_list)
    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / img_name
        image = Image.open(img_path).convert('RGB')
        image = image.resize((self.resolution, self.resolution), Image.BILINEAR)
        if self.labels is not None:
            return image, self.labels[idx], img_name
        return image, img_name
    
def collate_fn(batch):
    if len(batch[0]) == 3:  # train/val (image, label, filename)
        images = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        filenames = [item[2] for item in batch]
        return images, labels, filenames
    else:  # test (image, filename)
        images = [item[0] for item in batch]
        filenames = [item[1] for item in batch]
        return images, filenames
