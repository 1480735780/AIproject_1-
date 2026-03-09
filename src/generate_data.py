import argparse
import json
import os
import random
from pathlib import Path

from captcha.image import ImageCaptcha


def generate_split(output_dir: Path, total: int, length: int, charset: str, width: int, height: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = ImageCaptcha(width=width, height=height)

    for i in range(total):
        text = "".join(random.choice(charset) for _ in range(length))
        image = generator.generate_image(text)
        generator.create_noise_dots(image, '#000000', 3, 30)
        generator.create_noise_curve(image, '#000000')
        image.save(output_dir / f"{text}_{i}.jpg")


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    generate_split(
        Path(config["train_data_path"]),
        config["train_num"],
        config["captcha_length"],
        config["characters"],
        config["img_width"],
        config["img_height"],
    )
    generate_split(
        Path(config["test_data_path"]),
        config["test_num"],
        config["captcha_length"],
        config["characters"],
        config["img_width"],
        config["img_height"],
    )
    print("Data generation done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)
