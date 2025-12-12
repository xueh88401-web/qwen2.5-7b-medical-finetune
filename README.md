# Qwen2.5-7B 医学助手 (LoRA微调版) - 项目仓库

[![GitHub](https://img.shields.io/github/license/xueh88401-web/qwen2.5-7b-medical-finetune)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/🤗-Model%20on%20HF-yellow)](https://huggingface.co/xueh88401/qwen2.5-7b-medical)

> 此GitHub仓库包含了模型的LoRA适配器文件、训练配置及使用示例。模型主要托管在HuggingFace。

---
language:
  - zh
license: mit
tags:
  - medical
  - qwen
  - lora
  - fine-tuned
base_model: Qwen/Qwen2.5-7B-Instruct
datasets:
  - CMKRG/QiZhenGPT
---

# Qwen2.5-7B 医疗大模型微调

使用 QLoRA 技术在 **QiZhenGPT** 医疗数据集上对 Qwen2.5-7B-Instruct 进行指令微调，提升其在医疗问答中的专业性和准确性。

## 训练详情
- **方法**: QLoRA (4-bit NF4 + LoRA, r=8)
- **数据**: [QiZhenGPT](https://github.com/CMKRG/QiZhenGPT) (~20k 样本)
- **框架**: LLaMA-Factory
- **训练时长**: 6.8 小时 (6750 steps)
- **验证损失**: 1.578 (下降 23.8%)

## 使用说明
本仓库提供的是 LoRA 适配器权重，需要与基础模型 `Qwen/Qwen2.5-7B-Instruct` 结合使用。

### 使用 Transformers + PEFT 加载

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 加载基础模型
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)
# 加载本 LoRA 适配器
model = PeftModel.from_pretrained(model, "xueh88401/qwen2.5-7b-medical")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 进行推理
inputs = tokenizer("用户：感冒了怎么办？\n助手：", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

---
## 📁 本GitHub仓库文件说明

此仓库作为HuggingFace模型页面的补充，提供完整的项目文件以便复现和研究：
```
.
├── README.md                 # 本说明文件
├── inference_demo.py         # 简易推理脚本
├── adapter/                  # LoRA适配器权重与相关文件
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   └── ...
└── training/                 # 训练配置文件
    └── your_training_config.yaml
```
要使用模型，建议直接访问 [HuggingFace模型页面](https://huggingface.co/xueh88401/qwen2.5-7b-medical)。