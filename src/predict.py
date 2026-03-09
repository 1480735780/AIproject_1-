import argparse
import json

import torch
from PIL import Image
from torchvision import transforms

from charset import build_mappings
from model import CaptchaCNN


def decode(logits, idx2char):
    pred = torch.argmax(logits, dim=2).squeeze(0).tolist()
    return "".join(idx2char[i] for i in pred)


def main(config_path, model_path, image_path):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    _, idx2char = build_mappings(cfg["characters"])

    ckpt = torch.load(model_path, map_location="cpu")
    model = CaptchaCNN(num_chars=len(cfg["characters"]), captcha_length=cfg["captcha_length"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((cfg["resize_height"], cfg["resize_width"])),
        transforms.ToTensor(),
    ])

    image = Image.open(image_path).convert("L")
    x = tfm(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
    print(decode(logits, idx2char))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    main(args.config, args.model, args.image)
