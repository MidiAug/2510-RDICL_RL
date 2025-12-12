# RDICL_RL

Reinforcement Learning with In-Context Learning (RDICL) 项目

## 项目结构

```
.
├── script/          # 脚本文件
│   ├── data/       # 数据处理脚本
│   ├── stf/        # SFT (Supervised Fine-Tuning) 脚本
│   └── rl/         # Reinforcement Learning 脚本
├── data/           # 数据文件（不包含在git中）
├── ckpts/          # 模型检查点（不包含在git中）
├── logs/           # 日志文件（不包含在git中）
└── wandb/          # WandB实验跟踪（不包含在git中）
```

## 主要功能

- **数据处理**: 从训练数据中提取ICL演示样本
- **SFT训练**: 监督微调脚本
- **RL训练**: 强化学习训练脚本

## 使用说明

### 数据处理

1. 提取ICL演示样本：
```bash
python script/data/extract-icl_demo.py
```

2. 创建ICL数据集：
```bash
python script/data/create_icl_dataset.py
```

### 训练

参考 `script/stf/` 和 `script/rl/` 目录下的脚本进行训练。

## 依赖

安装依赖：
```bash
pip install -r requirements.txt
```

