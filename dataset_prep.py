import torch
from datasets import load_dataset
from torchvision.transforms import v2 
from torch.utils.data import Dataset
from PIL import Image

def is_rgb_image(example):
    # Check if the image mode is RGB
    return example['image'].mode == 'RGB'

# Apply the filter

class ImageNetDataset(Dataset):
    def __init__(self, num_classes, dtype = torch.float32, transform = None):
        self.dtype = dtype
        self.num_classes = num_classes
        super().__init__()
        self.hfds = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Compose([
            
            v2.Resize((224, 224)),  # Resize the image to 224x224 (standard for ImageNet)
            v2.RandomHorizontalFlip(p=0.5),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            v2.RandomPerspective(distortion_scale=0.5, p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
            v2.ToImage(), v2.ToDtype(dtype, scale=True)  # Convert the image to a PyTorch tensor
            ])
        self.normalization = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.hfds["train"])

    def __getitem__(self, index):
        datapoint = self.hfds["train"][index]
        pil_image = datapoint["image"]
        pil_image = pil_image.convert("RGB")

        y_number = datapoint["label"] # this is a single number ie. 539
        x_tensor = self.transform(pil_image).to(self.dtype)

        x_tensor = self.normalization(x_tensor)

        y_vector = torch.zeros((self.num_classes), dtype = self.dtype)
        y_vector[y_number] = 1 # set the corresponding position to 1
        return x_tensor, y_vector
    


class ValidationImageNetDataset(Dataset):
    def __init__(self, num_classes, dtype = torch.float32, transform = None):
        self.dtype = dtype
        self.num_classes = num_classes
        super().__init__()
        self.hfds = load_dataset("ILSVRC/imagenet-1k", trust_remote_code=True)
        if transform is not None:
            self.transform = transform
        else:
            self.transform = v2.Compose([
            
            v2.Resize((224, 224)),  # Resize the image to 224x224 (standard for ImageNet)
            v2.ToImage(), v2.ToDtype(dtype, scale=True)  # Convert the image to a PyTorch tensor
            ])
        self.normalization = v2.Compose([v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.hfds["validation"])

    def __getitem__(self, index):
        datapoint = self.hfds["validation"][index]
        pil_image = datapoint["image"]
        pil_image = pil_image.convert("RGB")
        
        y_number = datapoint["label"] # this is a single number ie. 539
        x_tensor = self.transform(pil_image).to(self.dtype)

        x_tensor = self.normalization(x_tensor)

        y_vector = torch.zeros((self.num_classes), dtype = self.dtype)
        y_vector[y_number] = 1 # set the corresponding position to 1
        return x_tensor, y_vector

'''x = ValidationImageNetDataset(1000)
x[0]'''