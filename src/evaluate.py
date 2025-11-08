# src/evaluate.py
import os
import argparse
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import SimpleVGG  # uses the SimpleVGG in src/model.py

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate VGG-like model on test set")
    p.add_argument("--test-dir", type=str, default="casting_data/casting_data/test", help="Path to test dataset (ImageFolder)")
    p.add_argument("--model-path", type=str, default="models/vgg_epoch3.pth", help="Path to saved model .pth")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()

def evaluate(test_dir, model_path, batch_size=16, image_size=224, num_workers=0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Test directory not found: {test_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()
    ])

    test_ds = datasets.ImageFolder(test_dir, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = test_ds.classes
    num_classes = len(classes)
    print("Classes (index -> name):", list(enumerate(classes)))

    # load model
    model = SimpleVGG(in_channels=3, num_classes=num_classes).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # confusion matrix
    conf_mat = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_mat[t.item()][p.item()] += 1
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    overall_acc = (correct / total) if total > 0 else 0.0
    print(f"\nTotal samples: {total}")
    print(f"Overall accuracy: {overall_acc:.4f}\n")

    # per-class accuracy
    print("Per-class accuracy:")
    for idx, cls_name in enumerate(classes):
        true_count = sum(conf_mat[idx])
        correct_count = conf_mat[idx][idx]
        acc = (correct_count / true_count) if true_count > 0 else 0.0
        print(f"  {idx:2d} {cls_name:20s} - {correct_count}/{true_count} = {acc:.4f}")

    # print confusion matrix
    print("\nConfusion matrix (rows: true class, cols: predicted class):")
    header = "     " + " ".join([f"{i:6d}" for i in range(num_classes)])
    print(header)
    for i, row in enumerate(conf_mat):
        row_str = " ".join(f"{c:6d}" for c in row)
        print(f"{i:3d} | {row_str}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(
        test_dir=args.test_dir,
        model_path=args.model_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers
    )
