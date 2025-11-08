# src/model.py
import torch
import torch.nn as nn

class SimpleVGG(nn.Module):
    """
    Lightweight VGG-like model:
      - small number of filters per block
      - 3 conv blocks (2 conv layers each) + pooling
      - adaptive pooling to make it input-size tolerant
      - small classifier
    Good for quick experiments / small datasets.
    """
    def __init__(self, in_channels=3, num_classes=2):
        super().__init__()
        # conv blocks (Conv -> ReLU -> Conv -> ReLU -> MaxPool)
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # adaptive pooling -> fixed small spatial size
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # small classifier (flatten -> fc -> relu -> dropout -> out)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 1 * 1, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x


# quick smoke test (run this file directly to check output shape)
if __name__ == "__main__":
    model = SimpleVGG(in_channels=3, num_classes=2)
    x = torch.randn(4, 3, 224, 224)   # batch of 4, 224x224 images
    out = model(x)
    print("Output shape:", out.shape)  # expect: (4, 2)
