import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from charset import build_mappings
from dataset import CaptchaDataset
from model import CaptchaCNN, batch_accuracy, captcha_loss


def evaluate(model, loader, device):
    model.eval()
    total_loss, total_char_acc, total_sample_acc = 0.0, 0.0, 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = captcha_loss(logits, labels)
            char_acc, sample_acc = batch_accuracy(logits, labels)
            total_loss += loss.item()
            total_char_acc += char_acc
            total_sample_acc += sample_acc

    n = len(loader)
    return total_loss / n, total_char_acc / n, total_sample_acc / n


def main(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    char2idx, _ = build_mappings(cfg["characters"])

    transform = transforms.Compose([
        transforms.Resize((cfg["resize_height"], cfg["resize_width"])),
        transforms.ToTensor(),
    ])

    train_ds = CaptchaDataset(cfg["train_data_path"], transform, char2idx, cfg["captcha_length"])
    test_ds = CaptchaDataset(cfg["test_data_path"], transform, char2idx, cfg["captcha_length"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=0)

    model = CaptchaCNN(num_chars=len(cfg["characters"]), captcha_length=cfg["captcha_length"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])

    best_acc = -1.0
    Path(cfg["model_save_path"]).mkdir(parents=True, exist_ok=True)
    save_path = Path(cfg["model_save_path"]) / f"{cfg['model_name']}.pt"

    for epoch in range(1, cfg["epoch_num"] + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = captcha_loss(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, char_acc, sample_acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch:03d} | val_loss={val_loss:.4f} | char_acc={char_acc:.4f} | sample_acc={sample_acc:.4f}")

        if sample_acc > best_acc:
            best_acc = sample_acc
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "config": cfg,
                },
                save_path,
            )

    print(f"Training done. best_sample_acc={best_acc:.4f}, model={save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)
