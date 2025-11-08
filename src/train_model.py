# src/train.py
import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# import the VGG-like model (make sure src/model.py defines SimpleVGG)
from model import SimpleVGG

def get_loaders(train_dir, val_dir, image_size=224, batch_size=16):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    val_ds = datasets.ImageFolder(val_dir, transform=transform)

    # Windows safe: num_workers=0. On Linux/macOS you can set >0 for speed.
    num_workers = 0 if os.name == 'nt' else 4

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def train(train_dir, val_dir, epochs=10, batch_size=16, image_size=224, lr=1e-3):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)

    train_loader, val_loader = get_loaders(train_dir, val_dir, image_size=image_size, batch_size=batch_size)
    classes = train_loader.dataset.classes
    print("Classes (index -> name):", list(enumerate(classes)))

    num_classes = len(classes)
    model = SimpleVGG(in_channels=3, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    os.makedirs("models", exist_ok=True)

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch} [train]", leave=False)
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = running_loss / (len(train_loader) if len(train_loader)>0 else 1)
        print(f"Epoch {epoch}, Train Loss: {avg_loss:.4f}")

        # validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                preds = model(imgs).argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = (correct / total) if total > 0 else 0.0
        print(f"Epoch {epoch}, Val Accuracy: {acc:.4f}")

        # save checkpoint
        ckpt_path = os.path.join("models", f"vgg_epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

if __name__ == "__main__":
    # Update these paths if your dataset is located elsewhere
    train_dir = "casting_data/casting_data/train"
    val_dir = "casting_data/casting_data/test"

    # change hyperparams as needed
    train(train_dir, val_dir, epochs=5, batch_size=8, image_size=224, lr=1e-3)
