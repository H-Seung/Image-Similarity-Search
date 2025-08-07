import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.anomaly_detector_encoder import AutoEncoder
from tqdm.auto import tqdm

# config
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-3
class_name = "metal_nut"  # 사용자 입력
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# dataset (정상 이미지만)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])
dataset = datasets.ImageFolder(f"{class_name}/train", transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# model
model = AutoEncoder().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# train
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    # Wrap the dataloader with tqdm for a progress bar
    for imgs, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

torch.save(model.state_dict(), f"autoencoder_{class_name}.pth")
