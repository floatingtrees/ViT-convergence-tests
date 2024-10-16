from biased_vit import ViT
import torch
import torchvision.transforms as transforms
from PIL import Image
from scheduler import Scheduler
import time
from dataset_prep import ImageNetDataset, ValidationImageNetDataset
from torch.utils.data import DataLoader
import sys

model = ViT(
    image_size = (224, 224),
    patch_size = 14,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 8,
    mlp_dim = 1024,
    dropout = 0.0,
    emb_dropout = 0.0
)

total_params = 0
for i in model.parameters():
    total_params += i.numel()
print(total_params)
device = "cuda"

batch_size = 512 # 1024 works, but it doesn't make much of a difference
dataset = ImageNetDataset(num_classes=1000, dtype = torch.bfloat16)
dataloader = DataLoader(dataset, batch_size = batch_size,  num_workers=31, prefetch_factor = 5, shuffle = True, pin_memory = False)

validation_dataset = ValidationImageNetDataset(1000, dtype = torch.bfloat16)
validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, num_workers=31, prefetch_factor = 5, shuffle = False, pin_memory = False)

opacity_scheduler = Scheduler(100)
model.to(device).to(torch.bfloat16)
num_epochs = 100
optimizer = torch.optim.AdamW(model.parameters())
#checkpoint = torch.load("model_and_optimizer_obscured390.pth")
#model.load_state_dict(checkpoint["model_state_dict"])


#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])



def compute_top1_accuracy(predictions, targets):
    """
    Compute the Top-1 accuracy.

    :param predictions: Tensor of shape (batch_size, num_classes) containing model predictions (logits or probabilities).
    :param targets: Tensor of shape (batch_size, num_classes) containing one-hot encoded class labels.
    
    :return: Top-1 accuracy as a percentage.
    """
    
    # Convert one-hot encoded targets to class indices
    targets_indices = torch.argmax(targets, dim=1)
    
    # Get the predicted class index (index of max logit/probability) for each sample
    predicted_indices = torch.argmax(predictions, dim=1)
    
    # Compare predicted indices to actual targets and compute accuracy
    correct_predictions = torch.eq(predicted_indices, targets_indices).sum().item()
    
    # Compute accuracy as the number of correct predictions divided by total samples
    accuracy = correct_predictions / targets.size(0)
    
    return accuracy * 100  # Return accuracy as a percentage

loss_fn = torch.nn.CrossEntropyLoss()
total_training_steps = len(dataset) * num_epochs
running_losses = []
start = time.perf_counter()
for i in range(num_epochs):
    num_preprocessed_batches = i * len(dataset)
    with torch.no_grad():
        accuracies = []
        validation_losses = []
        for i, batch in enumerate(validation_dataloader):
            opacity = 0#opacity_scheduler.sample((num_preprocessed_batches) / total_training_steps)
            x, y = batch
            x = x.to(device)
            y = y.to(device)
            outputs = model(x, opacity, device = device, dtype = torch.bfloat16)
            loss = loss_fn(outputs, y)
            validation_losses.append(loss.item())
            accuracies.append(compute_top1_accuracy(outputs, y))
        mean_last_items = sum(validation_losses) / len(validation_losses)
        print("VALIDATION LOSS: ", mean_last_items)
        print("VALIDATION ACCURACY", sum(accuracies) / len(accuracies))
    #print(time.perf_counter() - start)

    for j, batch in enumerate(dataloader):
        optimizer.zero_grad()
        opacity = 0#opacity_scheduler.sample((num_preprocessed_batches + j * batch_size) / total_training_steps)
        #with open("opacities.txt", 'a') as file:
        #    file.write(f'{opacity}\n')
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        outputs = model(x, opacity, device = device, dtype = torch.bfloat16)
        loss = loss_fn(outputs, y)
        loss.backward()
        optimizer.step()
        running_losses.append(loss.item())
        if j % 250 == 0:
            last_items = running_losses[-1000:] if len(running_losses) >= 1000 else running_losses
            mean_last_items = sum(last_items) / len(last_items)
            print(mean_last_items, loss.item())
            sys.stdout.flush()
    save_path = f"model_and_optimizer_standard{i}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
    