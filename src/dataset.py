import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class CaptchaDataset(Dataset):
    def __init__(self, data_dir: str, transform, char2idx: dict, captcha_length: int):
        self.files = [str(Path(data_dir) / x) for x in os.listdir(data_dir) if x.lower().endswith((".jpg", ".png", ".jpeg"))]
        self.transform = transform
        self.char2idx = char2idx
        self.captcha_length = captcha_length

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        file_path = self.files[index]
        image = Image.open(file_path).convert("L")
        image = self.transform(image)

        label_str = os.path.basename(file_path).split("_")[0]
        if len(label_str) != self.captcha_length:
            raise ValueError(f"Label length mismatch in {file_path}: {label_str}")

        label = torch.tensor([self.char2idx[c] for c in label_str], dtype=torch.long)
        return image, label
