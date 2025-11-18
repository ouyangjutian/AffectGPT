# AU Agent集成使用指南

## 📋 概述

将AU Agent集成到AffectGPT，实现端到端的AU模态处理：
```
视频 → OpenFace AU检测 → AU Agent (Qwen2.5 + LoRA) → AU描述 → CLIP编码 → AU特征
```

---

## 🎯 完整流程

### **步骤1: 准备AU指令数据集**

使用MER-Factory已生成的AU描述数据：

```bash
cd /home/project/AffectGPT/AffectGPT

# 从MER-Factory输出构建指令数据集
python prepare_au_instruction_dataset.py
```

**生成文件**：
- `au_instruction_dataset.json` - 完整数据集
- `au_instruction_dataset.jsonl` - LLaMA-Factory格式

**数据格式示例**：
```json
{
  "instruction": "Based on the following Action Unit detections, describe the facial expression:",
  "input": "AU01: 0.98, AU05: 0.98, AU07: 2.35, AU25: 1.76",
  "output": "The facial expression exhibits subtle brow lowering, neutral ocular engagement with mild lid tightening, and slight lip parting, consistent with a prototypical neutral state."
}
```

---

### **步骤2: 微调AU Agent**

使用LLaMA-Factory微调Qwen2.5-7B：

```bash
# 安装LLaMA-Factory（如果还没有）
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .

# 返回AffectGPT目录
cd /home/project/AffectGPT/AffectGPT

# 开始训练
bash train_au_agent.sh
```

**训练参数**：
- **基础模型**: Qwen2.5-7B-Instruct
- **方法**: LoRA (rank=64, alpha=128)
- **Epochs**: 3
- **Batch Size**: 4 × 4 (gradient accumulation)
- **Learning Rate**: 5e-5
- **预计时间**: 8-12小时（单GPU，100K样本）

**输出**：
```
./output/au_agent_qwen2.5_7b_lora/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-best/  ← 使用这个
└── ...
```

---

### **步骤3: 测试AU Agent**

验证微调效果：

```bash
python test_au_agent.py
```

**测试输出示例**：
```
Test Case 1: Neutral Expression
AU Values: {'AU01': 0.98, 'AU05': 0.98, 'AU07': 2.35, 'AU25': 1.76'}

Generated Description:
  The facial expression exhibits subtle brow lowering, neutral ocular 
  engagement with mild lid tightening, and slight lip parting, 
  consistent with a prototypical neutral state.
```

---

### **步骤4: 修改base_dataset.py集成AU Agent**

在`base_dataset.py`的`__init__`中添加AU Agent初始化：

```python
# my_affectgpt/datasets/datasets/base_dataset.py

from my_affectgpt.models.au_agent import create_au_agent

class BaseDataset():
    def __init__(self, ..., model_cfg=None, dataset_cfg=None, ...):
        # ... 现有代码 ...
        
        # AU Agent配置
        self.use_au_agent = getattr(dataset_cfg, 'use_au_agent', False)
        if self.use_au_agent:
            self.au_agent = create_au_agent(dataset_cfg)
            print(f"[Dataset] AU Agent enabled")
        else:
            self.au_agent = None
```

修改`_extract_au_features_realtime`使用AU Agent：

```python
def _extract_au_features_realtime(self, video_name):
    """实时从OpenFace检测 + AU Agent生成描述 + CLIP编码"""
    
    if self.use_au_agent:
        # 方案A: 使用AU Agent生成描述（新方案）
        return self._extract_au_with_agent(video_name)
    else:
        # 方案B: 读取MER-Factory预生成的描述（原方案）
        return self._extract_au_from_json(video_name)

def _extract_au_with_agent(self, video_name):
    """使用AU Agent实时生成AU描述"""
    import pandas as pd
    
    # 1. 读取OpenFace CSV
    openface_csv = os.path.join(self.openface_output_dir, f"{video_name}.csv")
    if not os.path.exists(openface_csv):
        print(f"⚠️ OpenFace output not found: {openface_csv}")
        return None
    
    df = pd.read_csv(openface_csv)
    
    # 2. 为每一帧生成AU描述
    descriptions = []
    for idx, row in df.iterrows():
        au_values = self.au_agent.parse_openface_csv(row.to_dict())
        description = self.au_agent.generate_description(au_values)
        descriptions.append(description)
    
    # 3. 使用CLIP编码描述
    clip_model = self._load_clip_for_au()
    if clip_model is None:
        return None
    
    import clip
    device = next(clip_model.parameters()).device
    text_tokens = clip.tokenize(descriptions, truncate=True).to(device)
    
    with torch.no_grad():
        text_features = clip_model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    return text_features
```

---

### **步骤5: 配置训练使用AU Agent**

修改训练配置文件：

```yaml
# train_configs/config_with_au_agent.yaml

datasets:
  mercaptionplus:
    # ... 其他配置 ...
    
    # AU Agent配置
    use_au_agent: true  # 启用AU Agent
    au_agent_base_model: /home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct
    au_agent_lora_weights: /home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora/checkpoint-best
    au_agent_use_lora: true
    
    # OpenFace输出路径（如果使用AU Agent实时生成）
    openface_output_dir: /home/project/openface_outputs
```

---

### **步骤6: 训练AffectGPT with AU Agent**

```bash
# 训练
python train.py --cfg-path train_configs/config_with_au_agent.yaml
```

**训练流程**：
```
视频 → OpenFace检测AU
         ↓
    AU Agent生成描述
         ↓
    CLIP编码为特征 [T, 512]
         ↓
    Q-Former处理
         ↓
    投影到LLM空间
         ↓
    与其他模态融合
         ↓
    LLM生成回复
```

---

### **步骤7: 推理使用AU Agent**

修改`inference_hybird.py`：

```python
# 推理配置
if use_au:
    dataset_cls.use_au_agent = True
    dataset_cls.au_agent_base_model = "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
    dataset_cls.au_agent_lora_weights = "./output/au_agent_qwen2.5_7b_lora/checkpoint-best"
    dataset_cls.openface_output_dir = "/home/project/openface_outputs"
```

运行推理：

```bash
python inference_hybird.py --cfg-path inference_config_au_agent.yaml
```

---

## 📊 方案对比

### **原方案 vs AU Agent方案**

| 方案 | AU检测 | AU描述生成 | CLIP编码 | 优点 | 缺点 |
|------|--------|-----------|---------|------|------|
| **原方案（MER-Factory）** | OpenFace | GPT-4o/Gemini API | ✅ | 描述质量高 | 需要API费用 |
| **AU Agent方案（新）** | OpenFace | Qwen2.5 + LoRA | ✅ | 完全免费，可定制 | 需要微调 |

---

## 🎯 优势

### **1. 完全免费**
- ✅ 无需GPT-4o API费用（节省$768）
- ✅ 无需Gemini API费用（节省$23）
- ✅ 本地推理，无网络限制

### **2. 可定制**
- ✅ 针对情感识别任务微调
- ✅ 可以添加领域知识
- ✅ 可以调整描述风格

### **3. 性能**
- ✅ Qwen2.5-7B性能接近GPT-4o
- ✅ LoRA微调后更适合AU任务
- ✅ 推理速度快（本地GPU）

### **4. 数据隐私**
- ✅ 数据不离开本地
- ✅ 适合敏感数据

---

## ⏱️ 时间成本

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据准备 | 1-2小时 | 从MER-Factory提取100K样本 |
| AU Agent微调 | 8-12小时 | 单GPU，Qwen2.5-7B + LoRA |
| 测试验证 | 30分钟 | 测试生成质量 |
| 集成代码 | 2-3小时 | 修改base_dataset.py等 |
| **总计** | **~15小时** | 一次性工作 |

---

## 💾 显存需求

| 操作 | 显存 | 配置 |
|------|------|------|
| **训练AU Agent** | 24GB | LoRA, bf16, gradient checkpointing |
| **推理AU Agent** | 8GB | 仅推理，bf16 |
| **AffectGPT训练（含AU Agent）** | 40GB | 建议A100 |

**优化**：
- 使用int8量化：显存减半
- 使用DeepSpeed ZeRO：分布式训练

---

## 🔧 故障排查

### **问题1: AU Agent生成质量差**

**原因**：微调不充分

**解决**：
```bash
# 增加训练轮数
num_train_epochs: 5

# 或增加数据量
max_samples: 200000
```

---

### **问题2: 显存不足**

**解决**：
```bash
# 减小batch size
per_device_train_batch_size: 2

# 或使用int8量化
load_in_8bit: true
```

---

### **问题3: 推理速度慢**

**解决**：
```python
# 1. 批量生成
batch_size = 16
descriptions = au_agent.batch_generate_descriptions(au_values_list, batch_size)

# 2. 使用vLLM加速
from vllm import LLM
llm = LLM(model=au_agent_path)
```

---

## 📝 配置示例

### **完整训练配置**

```yaml
# train_configs/emercoarse_au_agent.yaml

model:
  face_or_frame: multiface_audio_face_frame_au_text
  # ... 其他模型配置 ...

datasets:
  mercaptionplus:
    face_or_frame: multiface_audio_face_frame_au_text
    
    # AU Agent配置
    use_au_agent: true
    au_agent_base_model: /home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct
    au_agent_lora_weights: ./output/au_agent_qwen2.5_7b_lora/checkpoint-best
    au_agent_use_lora: true
    openface_output_dir: /home/project/openface_outputs
    
    # 或者使用MER-Factory预生成的描述（原方案）
    # use_au_agent: false
    # mer_factory_output: /home/project/MER-Factory/output
```

---

## 🎉 总结

**AU Agent方案优势**：
1. ✅ **完全免费**（无API费用）
2. ✅ **可定制**（针对任务微调）
3. ✅ **高性能**（Qwen2.5-7B + LoRA）
4. ✅ **端到端**（与AffectGPT完美集成）

**适用场景**：
- 需要大规模AU处理（API费用太高）
- 需要定制AU描述风格
- 数据隐私敏感
- 离线环境

现在你可以像论文一样，使用AU Agent实现完整的AU模态处理！🎊
