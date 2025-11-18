# AU Agent 微调工具包

本文件夹包含AU Agent的完整微调流程，用于训练Qwen2.5-7B-Instruct模型，实现从AU检测值到自然语言描述的转换。

---

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| **prepare_au_instruction_dataset.py** | 准备训练数据集（从MER-Factory提取） |
| **train_au_agent.sh** | AU Agent微调脚本（使用LLaMA-Factory） |
| **test_au_agent.py** | 测试AU Agent生成质量 |
| **au_agent_lora_config.yaml** | LoRA微调配置文件 |
| **setup_au_agent.sh** | 一键设置脚本（自动化全流程） |
| **AU_AGENT_INTEGRATION_GUIDE.md** | 详细集成指南 |
| **AU_AGENT_SUMMARY.md** | 方案总结 |

---

## 🚀 快速开始

### **方式1: 一键完成（推荐）**

```bash
cd /home/project/AffectGPT/AffectGPT/au_agent_finetune

# 执行一键设置（数据准备 + 训练 + 测试）
bash setup_au_agent.sh
```

---

### **方式2: 分步执行**

#### **步骤1: 准备数据集**

```bash
cd /home/project/AffectGPT/AffectGPT/au_agent_finetune

python prepare_au_instruction_dataset.py
```

**输出**：
- `au_instruction_dataset.json` - 完整数据集
- `au_instruction_dataset.jsonl` - LLaMA-Factory格式

---

#### **步骤2: 微调AU Agent**

```bash
bash train_au_agent.sh
```

**训练参数**：
- 基础模型：Qwen2.5-7B-Instruct
- 方法：LoRA (rank=64, alpha=128)
- 训练轮数：3 epochs
- 预计时间：8-12小时（单GPU）

**输出**：
```
../output/au_agent_qwen2.5_7b_lora/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-best/  ← 使用这个
└── ...
```

---

#### **步骤3: 测试AU Agent**

```bash
python test_au_agent.py
```

**功能**：验证生成质量，测试不同AU组合

---

## 📊 使用场景

### **场景1: 训练时实时生成AU描述（推荐）**

在训练配置中启用AU Agent：

```yaml
# train_configs/config_with_au_agent.yaml

datasets:
  mercaptionplus:
    face_or_frame: multiface_audio_face_frame_au_text
    
    # 启用AU Agent
    use_au_agent: true
    au_agent_base_model: /home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct
    au_agent_lora_weights: ../output/au_agent_qwen2.5_7b_lora/checkpoint-best
    openface_output_dir: /home/project/openface_outputs
```

**训练流程**：
```
视频 → OpenFace → AU Agent → CLIP → AffectGPT训练
```

---

### **场景2: 推理时实时生成AU描述**

在推理配置中启用AU Agent：

```yaml
# inference_config_au_agent.yaml

inference:
  use_au_agent: true
  au_agent_base_model: /home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct
  au_agent_lora_weights: ../output/au_agent_qwen2.5_7b_lora/checkpoint-best
  openface_output_dir: /home/project/openface_outputs
```

**推理流程**：
```
测试视频 → OpenFace → AU Agent → CLIP → AffectGPT推理
```

---

## 🎯 与论文对比

| 项目 | 论文EmoChat | 本实现 |
|------|------------|--------|
| 基础模型 | LLaMA-3.2-1B | **Qwen2.5-7B** ✅ |
| 微调方法 | LoRA | LoRA ✅ |
| 数据来源 | GPT-4o生成 | MER-Factory/本地 ✅ |
| 成本 | API费用 | **完全免费** ✅ |
| 集成度 | 独立模块 | **AffectGPT集成** ✅ |

---

## 💰 成本对比

| 方案 | 设置成本 | 推理成本（64K视频） |
|------|---------|-------------------|
| GPT-4o API | $0 | $768 |
| Gemini API | $0 | $23 |
| **AU Agent** | **15小时** | **$0** |

---

## 📖 详细文档

- **集成指南**：查看 `AU_AGENT_INTEGRATION_GUIDE.md`
- **方案总结**：查看 `AU_AGENT_SUMMARY.md`
- **配置文件**：参考 `au_agent_lora_config.yaml`

---

## ⚙️ 配置说明

### **关键路径配置**

编辑 `prepare_au_instruction_dataset.py`：

```python
# 第180-182行
MER_FACTORY_OUTPUT = '/home/project/MER-Factory/output'  # ← 修改为你的路径
OUTPUT_JSON = './au_instruction_dataset.json'
OUTPUT_JSONL = './au_instruction_dataset.jsonl'
```

编辑 `train_au_agent.sh`：

```bash
# 第6-8行
BASE_MODEL="/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"  # ← 修改
DATASET_PATH="./au_instruction_dataset.jsonl"
OUTPUT_DIR="../output/au_agent_qwen2.5_7b_lora"  # ← 修改
```

---

## 🔧 故障排查

### **问题1: 找不到MER-Factory输出**

**错误**：`MER-Factory output not found`

**解决**：
```bash
# 检查路径
ls /home/project/MER-Factory/output

# 修改配置
vim prepare_au_instruction_dataset.py  # 更新 MER_FACTORY_OUTPUT
```

---

### **问题2: 显存不足**

**错误**：`CUDA out of memory`

**解决**：
```yaml
# 编辑 au_agent_lora_config.yaml
per_device_train_batch_size: 2  # 从4减到2
gradient_accumulation_steps: 8  # 从4增到8
```

---

### **问题3: LLaMA-Factory未安装**

**错误**：`LLaMA-Factory not found`

**解决**：
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git /home/project/LLaMA-Factory
cd /home/project/LLaMA-Factory
pip install -e .
```

---

## 📊 预期输出

### **数据集统计**

```
Total samples collected: 100,000+
Train: 95,000
Val: 5,000
```

### **训练日志**

```
Epoch 1/3: 100%|██████████| 5000/5000 [2:30:00<00:00]
Loss: 0.45
Eval Loss: 0.32
```

### **测试结果**

```
Test Case 1: Neutral Expression
AU Values: {'AU01': 0.98, 'AU05': 0.98, 'AU07': 2.35}
Generated: "The facial expression exhibits subtle brow lowering..."
```

---

## 🎉 使用流程

```bash
# 1. 进入微调文件夹
cd /home/project/AffectGPT/AffectGPT/au_agent_finetune

# 2. 一键设置
bash setup_au_agent.sh

# 3. 返回AffectGPT根目录
cd ..

# 4. 配置训练
vim train_configs/config_with_au_agent.yaml

# 5. 开始训练
python train.py --cfg-path train_configs/config_with_au_agent.yaml
```

---

## 📞 支持

遇到问题？
1. 查看 `AU_AGENT_INTEGRATION_GUIDE.md`
2. 检查路径配置
3. 查看训练日志：`../output/au_agent_qwen2.5_7b_lora/`

---

## 📝 版本信息

- **创建日期**：2025-11-17
- **基础模型**：Qwen2.5-7B-Instruct
- **微调方法**：LoRA (rank=64, alpha=128)
- **参考论文**：EmoChat (AAAI 2025)

---

祝微调顺利！🚀
