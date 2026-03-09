import torch
import torch.nn as nn


class CaptchaCNN(nn.Module):
    def __init__(self, num_chars: int, captcha_length: int):
        super().__init__()
        self.num_chars = num_chars
        self.captcha_length = captcha_length

        self.backbone = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, captcha_length * num_chars),
        )

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        return logits.view(-1, self.captcha_length, self.num_chars)


def captcha_loss(logits, targets):
    loss_fn = nn.CrossEntropyLoss()
    losses = [loss_fn(logits[:, i, :], targets[:, i]) for i in range(logits.size(1))]
    return torch.stack(losses).mean()


def batch_accuracy(logits, targets):
    pred = torch.argmax(logits, dim=2)
    char_acc = (pred == targets).float().mean().item()
    sample_acc = (pred == targets).all(dim=1).float().mean().item()
    return char_acc, sample_acc
