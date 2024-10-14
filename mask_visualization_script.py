from vit_visualization import ViT
import torch
import torchvision.transforms as transforms
from PIL import Image

# Load the image
image_path = "tester.jpg"
original_image = Image.open(image_path)

# Define the transformation pipeline
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

# Apply the transformation
transformed_tensor = transform(original_image)


# Define the inverse transformation
inverse_transform = transforms.Compose([
    transforms.ToPILImage(),
])
model = ViT(
    image_size = (512, 512),
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 3,
    heads = 8,
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
model(transformed_tensor)
