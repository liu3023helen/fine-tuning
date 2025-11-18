# Qwen3 Fine-tuning Project

## 项目简介

这是一个基于Qwen3模型的微调项目，使用LoRA（Low-Rank Adaptation）技术进行参数高效微调。项目包含数据处理、模型训练和推理的完整流程。

## 功能特性

- ✨ 基于Qwen3-0.6B模型的LoRA微调
- 🛠️ 自动化数据处理和转换
- 📊 支持中英文自我认知数据
- 🚀 使用MS-Swift框架进行高效训练
- 💾 支持模型检查点保存和恢复

## 项目结构

```
├── transform_data.py          # 数据转换脚本
├── terminal.sh               # 训练和推理命令
├── self_cognition.jsonl      # 原始数据集
├── self_cognition_futureai.jsonl  # 转换后的训练数据
├── requirements.txt          # 项目依赖
├── .gitignore               # Git忽略文件
└── README.md               # 项目文档
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 数据准备

首先准备你的自我认知数据集格式，然后运行数据转换：

```bash
python transform_data.py \
    --name_zh 小新 \
    --author_zh FutureAI实验室 \
    --name_en Xiao-xin \
    --author_en FutureAILab
```

### 3. 下载基础模型

```bash
modelscope download --model Qwen/Qwen3-0.6B --local_dir ./models/Qwen/Qwen3-0.6B
```

### 4. 开始训练

```bash
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model ./models/Qwen/Qwen3-0.6B \
    --train_type lora \
    --dataset './self_cognition_futureai.jsonl' \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 16 \
    --eval_step 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir ./outputs \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

### 5. 推理测试

训练完成后，使用以下命令进行推理：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters ./outputs/checkpoint-XXX \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048
```

## 核心依赖

- **PyTorch** - 深度学习框架
- **Transformers** - Hugging Face模型库
- **ModelScope** - 模型下载和管理
- **MS-Swift** - 高效微调框架
- **PEFT** - 参数高效微调
- **LoRA** - 低秩适应技术

## 数据格式

项目支持JSONL格式的数据集，格式如下：

```json
{"conversations": [{"from": "human", "value": "你是谁？"}, {"from": "gpt", "value": "我是{{NAME}}，由{{AUTHOR}}开发的AI助手。"}], "tag": "zh"}
```

## 主要参数说明

- **lora_rank**: LoRA矩阵的秩，影响参数量和性能
- **lora_alpha**: LoRA的缩放参数，控制更新强度
- **learning_rate**: 学习率，建议1e-4
- **warmup_ratio**: 预热比例，建议0.05
- **max_length**: 输入序列最大长度

## 注意事项

1. 确保GPU内存充足（建议8GB+）
2. 训练前检查数据集格式正确
3. 根据硬件调整batch_size和gradient_accumulation_steps
4. 建议设置save_total_limit来限制保存的检查点数量

## 许可证

本项目仅供学习和研究使用。

## 贡献

欢迎提交Issue和Pull Request来改进这个项目！

## 联系方式

如有问题，请通过GitHub Issues联系我们。