系统镜像
ubuntu22.04-cuda12.4.0-py311-torch2.8.0-1.31.0-LLM
# https://www.modelscope.cn/my/mynotebook


处理训练集数据
python transform_data.py --name_zh 小新 --author_zh HelenAI实验室 --name_en Xiao-xin --author_en HelenAILab


安装框架
pip install 'ms-swift'
安装依赖
pip install modelscope


下载基础模型
modelscope download --model Qwen/Qwen3-0.6B --local_dir /mnt/workspace/models/Qwen/Qwen3-0.6B


# 启动LoRA训练(基于 Swift 框架的模型微调命令)
# 设置环境变量：指定使用第0号GPU进行训练
CUDA_VISIBLE_DEVICES=0 \
# Swift框架的监督微调命令
swift sft \
    # 指定要微调的模型路径 (使用花括号表示需要替换的变量)
    --model {model_path} \
    # 指定训练类型为LoRA (低秩适应)
    --train_type lora \
    # 指定训练数据集文件路径 (JSON Lines格式)
    --dataset '{dataset_file}' \
    # 使用bfloat16数据类型 (节省显存，提高训练速度)
    --torch_dtype bfloat16 \
    # 训练轮数 (1个epoch)
    --num_train_epochs 1 \
    # 每个GPU的训练批次大小
    --per_device_train_batch_size 1 \
    # 每个GPU的评估批次大小
    --per_device_eval_batch_size 1 \
    # 学习率 (0.0001)
    --learning_rate 1e-4 \
    # LoRA矩阵的秩 (rank值，影响参数量和性能)
    --lora_rank 8 \
    # LoRA的缩放参数 (控制更新强度)
    --lora_alpha 32 \
    # 目标模块：将LoRA应用到所有线性层
    --target_modules all-linear \
    # 梯度累积步数 (有效批次大小 = 1 × 16 = 16)
    --gradient_accumulation_steps 16 \
    # 评估步数：每50步进行一次评估
    --eval_step 50 \
    # 保存步数：每50步保存一次模型
    --save_steps 50 \
    # 保存检查点数量限制：最多保存2个检查点
    --save_total_limit 2 \
    # 日志记录步数：每5步记录一次训练日志
    --logging_steps 5 \
    # 输入序列的最大长度 (限制token数量)
    --max_length 2048 \
    # 输出目录：模型和日志保存路径
    --output_dir output \
    # 系统提示词：定义模型的角色和行为
    --system 'You are a helpful assistant.' \
    # 预热比例：前5%的训练步骤用于学习率预热
    --warmup_ratio 0.05 \
    # 数据加载工作进程数 (提高数据加载效率)
    --dataloader_num_workers 4

export CUDA_VISIBLE_DEVICES=0
swift sft \
    --model /mnt/workspace/models/Qwen/Qwen3-0.6B \
    --train_type lora \
    --dataset '/mnt/workspace/ms-swift-demo/self_cognition_futureai.jsonl' \
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
    --output_dir /mnt/workspace/ms-swift-demo/outputs \
    --system 'You are a helpful assistant.' \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4

# ms-swift-demo/outputs/v0-20251118-164642/checkpoint-35/trainer_state.json


微调后推理
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters /mnt/workspace/ms-swift-demo/outputs/v0-20251118-164642/checkpoint-35 \
    --stream true \
    --temperature 0 \
    --max_new_tokens 2048











从您的 `terminal.sh` 文件可以看出，这个项目需要：

### 1. 必需的深度学习框架
```bash
# PyTorch - 必需的深度学习框架
pip install torch torchvision torchaudio

# Transformers - Hugging Face的transformer库
pip install transformers

# Datasets - 数据集处理库
pip install datasets

# Accelerate - 分布式训练加速
pip install accelerate

# PEFT - 参数高效微调库（LoRA等）
pip install peft
```

### 2. 模型相关库
```bash
# ModelScope - 您已经计划的
pip install modelscope

# MS-Swift - 您已经计划的
pip install 'ms-swift'

# BitsAndBytes - 量化支持
pip install bitsandbytes
```

### 3. 数据处理库
```bash
# 基础数据处理
pip install pandas numpy
pip install jsonlines  # 用于处理.jsonl文件
```

## 完整的安装命令应该是：

```bash
# 激活虚拟环境
Qwen3_fine_tuning_venv\Scripts\activate

# 1. 安装深度学习框架
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 2. 安装核心ML库
pip install transformers datasets accelerate

# 3. 安装微调相关库
pip install peft bitsandbytes

# 4. 安装ModelScope和Swift
pip install modelscope
pip install 'ms-swift'

# 5. 安装数据处理库
pip install pandas numpy jsonlines

# 6. 安装其他有用工具
pip install jupyter ipython
```

## 为什么只安装两个命令不够：

1. **运行时错误**：运行您的训练命令时会因为缺少依赖而报错
2. **功能缺失**：模型下载、训练、推理都需要多个库协同工作
3. **性能问题**：没有适当的加速库，训练会很慢

**建议**：按照完整的依赖列表安装，否则您的项目无法正常运行。