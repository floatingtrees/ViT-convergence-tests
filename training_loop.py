from biased_vit import ViT
import torch
import torchvision.transforms as transforms
from PIL import Image
from scheduler import Scheduler
import time
from dataset_prep import ImageNetDataset, ValidationImageNetDataset
from torch.utils.data import DataLoader


model = ViT(
    image_size = (224, 224),
    patch_size = 14,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.0,
    emb_dropout = 0.0
)

total_params = 0
for i in model.parameters():
    total_params += i.numel()

device = "cuda"

batch_size = 128
dataset = ImageNetDataset(num_classes=1000, dtype = torch.bfloat16)
dataloader = DataLoader(dataset, batch_size = batch_size,  num_workers=7, prefetch_factor = 5, shuffle = False)

validation_dataset = ValidationImageNetDataset(1000, dtype = torch.bfloat16)
validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size,  num_workers=7, prefetch_factor = 5, shuffle = False)

opacity_scheduler = Scheduler(100)
model.to(device).to(torch.bfloat16)
num_epochs = 2






loss_fn = torch.nn.CrossEntropyLoss()
total_training_steps = len(dataset) * num_epochs
optimizer = torch.optim.AdamW(model.parameters())
running_losses = []
for i in range(num_epochs):
    num_preprocessed_batches = i * len(dataset)
    
    for j, batch in enumerate(dataloader):
        optimizer.zero_grad()
        opacity = opacity_scheduler.sample((num_preprocessed_batches + j * batch_size) / total_training_steps)
        with open("opacities.txt", 'a') as file:
            file.write(f'{opacity}\n')
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x, opacity, device = device, dtype = torch.bfloat16)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        running_losses.append(loss.item())
        if j % 100 == 0:
            last_items = running_losses[-100:] if len(running_losses) >= 100 else running_losses
            mean_last_items = sum(last_items) / len(last_items)
            print(mean_last_items)
        save_path = "model_and_optimizer.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    with torch.no_grad():
        validation_losses = []
        for i, batch in enumerate(validation_dataloader):
            opacity = opacity_scheduler.sample((num_preprocessed_batches) / total_training_steps)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x, opacity, device = device, dtype = torch.bfloat16)
            loss = loss_fn(outputs, y)
            validation_losses.append(loss)
        mean_last_items = sum(validation_losses) / len(validation_losses)
        print("VALIDATION LOSS: ", mean_last_items)