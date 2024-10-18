from biased_vit import ViT
import torch
from torchvision.transforms import v2 
from PIL import Image
from scheduler import Scheduler
import time
from dataset_prep import ImageNetDataset, ValidationImageNetDataset
from torch.utils.data import DataLoader
import sys
from torch.optim.lr_scheduler import LambdaLR

model = ViT(
    image_size = (224, 224),
    patch_size = 14,
    num_classes = 1000,
    dim = 384,
    depth = 12,
    heads = 6,
    mlp_dim = 1536,
    dropout = 0.03,
    emb_dropout = 0.01
)

total_params = 0
for i in model.parameters():
    total_params += i.numel()
print(total_params)
device = "cuda"

batch_size = 256 # 1024 works, but it doesn't make much of a difference
dataset = ImageNetDataset(num_classes=1000, dtype = torch.bfloat16)
dataloader = DataLoader(dataset, batch_size = batch_size,  num_workers=31, prefetch_factor = 5, shuffle = True, pin_memory = False)

validation_dataset = ValidationImageNetDataset(1000, dtype = torch.bfloat16)
validation_dataloader = DataLoader(validation_dataset, batch_size = batch_size, num_workers=31, prefetch_factor = 5, shuffle = False, pin_memory = False)

opacity_scheduler = Scheduler(100)
model.to(device).to(torch.bfloat16)
start_epoch = 0
num_epochs = 300
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001, betas = (0.9, 0.999),weight_decay = 0.03)
warmup_epochs = 35
import math
def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return (float(epoch) + 1) / (float(warmup_epochs) + 1)
    else:
        return math.cos(((epoch - warmup_epochs) / (num_epochs - warmup_epochs)) * math.pi/2)
    

#checkpoint = torch.load("model_and_optimizer_low_learning_rate27.pth")
#model.load_state_dict(checkpoint["model_state_dict"])
#optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)


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


augmentation_transform = v2.Compose([
            v2.RandomHorizontalFlip(p=0.5),
            v2.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
            v2.RandomAffine(degrees=(30, 70), translate=(0.1, 0.3), scale=(0.5, 0.75)),
            v2.RandomPerspective(distortion_scale=0.5, p=0.5),
            v2.RandomRotation(degrees=(0, 180)),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])



start = time.perf_counter()
for epoch in range(num_epochs):
    if epoch < start_epoch:
        scheduler.step()
        print(optimizer.param_groups[0]["lr"])
        continue
    num_preprocessed_batches = epoch * len(dataset)
    
    #print(time.perf_counter() - start)

    for j, batch in enumerate(dataloader):
        optimizer.zero_grad()
        opacity = 0#opacity_scheduler.sample((num_preprocessed_batches + j * batch_size) / total_training_steps)
        #with open("opacities.txt", 'a') as file:
        #    file.write(f'{opacity}\n')
        x, y = batch
        x = x.to(device)
        y = y.to(device)
        x = augmentation_transform(x)
        outputs = model(x, opacity, device = device, dtype = torch.bfloat16)
        loss = loss_fn(outputs, y)
        loss.backward()
        clip_value = 1.0
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
        optimizer.step()
        
        
        running_losses.append(loss.item())
        if j % 250 == 249:
            last_items = running_losses[-1000:] if len(running_losses) >= 1000 else running_losses
            mean_last_items = sum(last_items) / len(last_items)
            print(mean_last_items, loss.item())
            sys.stdout.flush()
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])
    save_path = f"model_and_optimizer_deep{epoch}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, save_path)
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
    