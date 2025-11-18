# AU Agent 集成说明

## ✅ 已完成的修改

### 1. **AU Agent 模型** (`au_agent.py`)
- ✅ 支持从AU值生成自然语言描述
- ✅ 优先使用外部提供的`au_description`（来自MER-Factory）
- ✅ 自动回退机制（如果没有外部描述则自动生成）
- ✅ **推理时只使用AU result**（不包含Emotion和Prompt）
- ✅ **生成客观的肌肉运动描述**（无情感词）

### 2. **数据加载** (`base_dataset.py`)
- ✅ 添加`_load_au_result_from_mer_factory()`方法
- ✅ 从MER-Factory JSON加载AU result
- ✅ 返回格式：`{'active_aus': {...}, 'au_description': "..."}`
- ✅ 支持峰值帧和均匀采样策略

### 3. **推理处理** (`conversation_video.py`)
- ✅ 添加AU Agent初始化
- ✅ `postprocess_au()`支持多种输入格式
- ✅ 自动调用AU Agent生成Facial Content描述
- ✅ 将描述转换为text tokens输入AffectGPT

### 4. **训练配置** (`train_configs/*.yaml`)
- ✅ 添加`mer_factory_output`路径配置
- ✅ 添加`use_au_agent`开关（训练和推理都设为True）
- ✅ 添加AU Agent模型路径配置

---

## 📊 完整数据流

### **重要说明：微调vs推理的输入格式**

**微调AU Agent时**：
```
输入: Emotion + Prompt + AU values + AU descriptions
目的: 让模型学习多种输入组合到描述的映射
```

**AffectGPT调用AU Agent时**：
```
输入: AU values + AU descriptions (只有AU result)
目的: 生成客观的肌肉运动描述（Facial Content）
原因:
  - AU Agent只负责AU→描述的转换
  - 情感推理是AffectGPT的任务
  - 描述应该客观，不包含情感词
```

### **AffectGPT训练阶段**
```
1. MER-Factory生成AU result (OpenFace only, 不需要GPT-4o)
   └── {sample_name}_au_analysis.json

2. base_dataset.py加载
   └── _load_au_result_from_mer_factory()
   └── 返回: {'active_aus': {...}, 'au_description': "..."}

3. conversation_video.py处理
   └── postprocess_au() 使用AU Agent (use_au_agent: True)
   └── AU Agent生成Facial Content描述
       输入: AU values + AU descriptions (只有AU result)
       输出: 客观的肌肉运动描述（无情感词）
   └── 转换为text tokens
   └── 输入AffectGPT训练
```

### **AffectGPT推理阶段**
```
1. MER-Factory生成AU result (相同)

2. base_dataset.py加载 (相同)

3. conversation_video.py处理 (相同)
   └── postprocess_au() 使用AU Agent (use_au_agent: True)
   └── AU Agent生成描述
       输入: AU values + AU descriptions (只有AU result)
       输出: 客观的肌肉运动描述（无情感词）
   └── 转换为text tokens
   └── 输入AffectGPT生成最终输出
```

---

## 🚀 使用步骤

### **步骤1: 为所有数据集生成AU result**

使用MER-Factory批量处理：

```bash
cd /home/project/MER-Factory

# 训练数据集（mercaptionplus）
python batch_process_au_only.py \
    --dataset mercaptionplus \
    --video_dir /path/to/videos \
    --output_dir ./output

# 推理数据集（9个）
for dataset in MER2023 MER2024 MELD IEMOCAP MOSI MOSEI SIMS SIMSv2 OVMERDPlus
do
    python batch_process_au_only.py \
        --dataset $dataset \
        --video_dir /path/to/${dataset}/videos \
        --output_dir ./output
done
```

**注意**：只需要OpenFace，不需要调用GPT-4o！

### **步骤2: 修改配置文件**

训练配置 (`train_configs/*.yaml`):
```yaml
model:
  use_au_agent: True  # 训练时使用AU Agent
  au_agent_base_model: "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
  au_agent_lora_weights: "/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora"
  
datasets:
  mercaptionplus:
    mer_factory_output: '/home/project/MER-Factory/output'
```

推理配置 (`eval_configs/*.yaml`):
```yaml
model:
  use_au_agent: True  # 推理时也使用AU Agent
  au_agent_base_model: "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
  au_agent_lora_weights: "/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora"
  
datasets:
  mer2023:  # 或其他数据集
    mer_factory_output: '/home/project/MER-Factory/output'
```

### **步骤3: 训练/推理**

```bash
# 训练（使用AU Agent）
python train.py --cfg-path train_configs/xxx.yaml

# 推理（使用AU Agent）
python inference.py --cfg-path eval_configs/xxx.yaml
```

---

## 📁 目录结构

```
G:\Project\MER-Factory\output\
├── mercaptionplus\
│   ├── sample_00001998\
│   │   └── sample_00001998_au_analysis.json
│   ├── sample_00002000\
│   │   └── sample_00002000_au_analysis.json
├── MER2023\
│   ├── video_001\
│   │   └── video_001_au_analysis.json
├── MER2024\
├── MELD\
├── IEMOCAP\
├── MOSI\
├── MOSEI\
├── SIMS\
├── SIMSv2\
└── OVMERDPlus\
```

每个JSON包含：
```json
{
  "per_frame_au_descriptions": [
    {
      "frame": 104,
      "au_description": "Brow lowerer (intensity: 0.88), ...",
      "active_aus": {
        "AU04_r": 0.88,
        "AU10_r": 2.37
      }
    }
  ]
}
```

---

## 🔧 配置说明

### **MER-Factory输出路径**
- `mer_factory_output`: MER-Factory生成的AU分析JSON文件根目录
- 期望路径：`{mer_factory_output}/{sample_name}/{sample_name}_au_analysis.json`

### **AU Agent开关**
- 训练时：`use_au_agent: True`（使用AU Agent生成Facial Content描述）
- 推理时：`use_au_agent: True`（使用AU Agent生成Facial Content描述）

### **采样策略**
- `frame_sampling: 'uniform'`: 使用第一帧的AU result
- `frame_sampling: 'emotion_peak'`: 使用峰值帧的AU result（如果有）

---

## ⚠️ 注意事项

1. **训练和推理都需要AU Agent**
   - 训练时：使用AU Agent生成Facial Content描述输入AffectGPT训练
   - 推理时：同样使用AU Agent生成描述
   - 需要额外显存加载AU Agent模型（~14GB）

2. **MER-Factory只需要OpenFace**
   - 不需要调用GPT-4o（只在微调AU Agent时需要）
   - AffectGPT训练/推理时只需`active_aus`和`au_description`

3. **显存要求**
   - AffectGPT (7B): ~14GB
   - AU Agent (7B + LoRA): ~14GB
   - **总计**: ~28GB
   - 建议使用80GB A100或多卡训练

4. **兼容多种输入格式**
   - MER-Factory JSON格式（推荐）
   - 预提取CLIP特征（旧方式，兼容性保留）
   - 自动回退机制

---

## 🎯 总结

**训练流程**：
```
OpenFace → AU result → AU Agent → Facial Content → AffectGPT训练
```

**推理流程**：
```
OpenFace → AU result → AU Agent → Facial Content → AffectGPT推理
```

**关键点**：
- ✅ 训练和推理都使用AU Agent
- ✅ 训练和推理共用MER-Factory输出
- ✅ 不需要重复调用GPT-4o（只在微调AU Agent时需要）
- ✅ AU Agent生成客观的肌肉运动描述（无情感词）
- ✅ 完全符合论文架构设计

---

## 📞 问题排查

### 问题1: AU result加载失败
```
⚠️ AU result加载失败: sample_xxx
```
**解决**：检查`mer_factory_output`路径和JSON文件是否存在

### 问题2: AU Agent生成失败
```
⚠️ AU Agent生成失败: xxx
```
**解决**：
1. 检查`use_au_agent: True`
2. 检查AU Agent模型路径
3. 检查显存是否足够

### 问题3: AU模态无效
```
⚠️ AU特征无效，跳过样本: xxx
```
**解决**：确保MER-Factory已为该样本生成AU result JSON
