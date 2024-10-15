from biased_vit import ViT
import torch
import torchvision.transforms as transforms
from PIL import Image
from scheduler import Scheduler
import time

# Load the image
image_path = "tester.jpg"
original_image = Image.open(image_path)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Apply the transformation
transformed_tensor = transform(original_image)

model = ViT(
    image_size = (224, 224),
    patch_size = 14,
    num_classes = 1000,
    dim = 1024,
    depth = 3,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

sum = 0
for i in model.parameters():
    sum += i.numel()
print(sum)

transformed_tensor = transformed_tensor.unsqueeze(0)
print(transformed_tensor.shape)
outputs = model(transformed_tensor,mask_constant = float('inf'))
print(outputs.shape)

data = torch.zeros(1024, 3, 224, 224).to("cuda")
opacity_scheduler = Scheduler(100)
start = time.perf_counter()
model.to("cuda")
for i in range(100):
    opacity = opacity_scheduler.sample(i / 10000)
    outputs = model(data, opacity, "cuda")
print(time.perf_counter() - StopAsyncIteration)