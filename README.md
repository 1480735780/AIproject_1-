# 基于 PyTorch 的 0-9 + A-Z 字符验证码识别

这是一个可直接运行的字符型验证码识别项目，支持字符集：

- 数字：`0-9`
- 大写字母：`A-Z`

共 36 类字符，默认识别 4 位验证码。

## 1. 项目结构

```text
.
├── config.json
├── requirements.txt
├── src
│   ├── charset.py
│   ├── dataset.py
│   ├── generate_data.py
│   ├── model.py
│   ├── predict.py
│   └── train.py
└── data/
    ├── train/
    └── test/
```

## 2. 安装依赖

```bash
pip install -r requirements.txt
```

## 3. 生成验证码数据

```bash
python src/generate_data.py --config config.json
```

验证码文件名格式：`AB3D_12.jpg`，文件名前缀就是标签。

## 4. 训练模型

```bash
python src/train.py --config config.json
```

训练完成后会保存最优模型到：

- `checkpoints/captcha_alnum36.pt`

## 5. 单张图片预测

```bash
python src/predict.py --config config.json --model checkpoints/captcha_alnum36.pt --image data/test/XXXX_0.jpg
```

输出即预测字符串。

## 6. 可调参数

在 `config.json` 中可调整：

- 数据规模（`train_num`, `test_num`）
- 验证码长度（`captcha_length`）
- 图像尺寸与训练输入尺寸
- batch size、学习率、epoch

---
如果你愿意，我还可以继续帮你加：
1) TensorBoard 训练可视化；
2) ONNX 导出；
3) FastAPI 推理接口；
4) 变长验证码（CTC/CRNN）。
