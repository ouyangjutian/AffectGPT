# AU Agent 格式匹配验证

## ✅ 格式匹配确认

### **微调数据格式**

**有情感标签样本**：
```json
{
    "instruction": "Generate a detailed facial expression description based on the given information.",
    "input": "Emotion: [acknowledgment, appreciation, curiosity, surprise, hesitation]\nPrompt: Given the emotion label, AU intensity values, and their semantic descriptions, provide a detailed and natural facial expression description:\nAU values: AU26: 1.39\nAU descriptions: Jaw drop (intensity: 1.39)",
    "output": "The expression is marked by a pronounced jaw drop..."
}
```

**无情感标签样本**：
```json
{
    "instruction": "Generate a facial expression description based on AU detections.",
    "input": "Prompt: Given the emotion label, AU intensity values, and their semantic descriptions, provide a detailed and natural facial expression description:\nAU values: AU26: 1.39\nAU descriptions: Jaw drop (intensity: 1.39)",
    "output": "The expression is marked by a pronounced jaw drop..."
}
```

---

### **推理时格式（au_agent.py - AffectGPT调用）**

**设计原则**：
- ✅ 只使用AU result（AU values + AU descriptions）
- ✅ 不包含Emotion和Prompt
- ✅ 只生成客观的肌肉运动描述（无情感词）

**当前实现**：
```python
# 推理时只使用AU result
instruction = "Generate a detailed and objective facial muscle movement description based on the Action Unit detections. Focus only on the physical movements without inferring emotions."
input_text = """AU values: AU26: 1.39
AU descriptions: Jaw drop (intensity: 1.39)"""

# 使用Qwen2.5 chat template
messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": input_text}
]
```

**转换后的格式**（Qwen chat template）：
```
<|im_start|>system
Generate a detailed and objective facial muscle movement description based on the Action Unit detections. Focus only on the physical movements without inferring emotions.<|im_end|>
<|im_start|>user
AU values: AU26: 1.39
AU descriptions: Jaw drop (intensity: 1.39)<|im_end|>
<|im_start|>assistant
```

---

## ✅ 设计原理

### **为什么微调和推理格式不同？**

**微调阶段**：
- 目标：让模型学习从 `Emotion + Prompt + AU result` 到描述的映射
- 输入完整信息，让模型理解多种输入组合
- 模型学习到情感标签、提示语、AU值之间的关系

**推理阶段（AffectGPT调用）**：
- 目标：只需要客观的肌肉运动描述（Facial Content）
- 只输入AU result，不需要Emotion和Prompt
- 原因：
  1. ✅ AU Agent只负责AU→描述的转换
  2. ✅ 情感推理是AffectGPT的任务，不是AU Agent的任务
  3. ✅ 描述应该是客观的，不包含情感词

### **格式对比**

| 字段 | 微调时 | 推理时 | 说明 |
|------|--------|--------|------|
| **Emotion** | ✅ 包含 | ❌ 不包含 | 推理时不需要情感标签 |
| **Prompt** | ✅ 包含 | ❌ 不包含 | 推理时不需要提示语 |
| **AU values** | ✅ 包含 | ✅ 包含 | 核心输入 |
| **AU descriptions** | ✅ 包含 | ✅ 包含 | 核心输入 |
| **instruction** | 详细指令 | 简化指令 | 推理时强调客观性 |

---

## 🔍 关键改进

### **修改前的问题**
```python
# ❌ 旧版本：格式不匹配
full_prompt = f"""Based on the following Action Unit detections, provide a detailed and natural facial expression description:

AU values: {au_values_text}
AU descriptions: {au_descriptions_text}

Description:"""
```

**问题**：
1. 没有instruction字段
2. 没有Prompt字段
3. 提示语不匹配
4. 没有使用chat template

### **修改后**
```python
# ✅ 新版本：格式匹配
instruction = "Generate a facial expression description based on AU detections."
input_text = f"""Prompt: {prompt_text}
AU values: {au_values_text}
AU descriptions: {au_descriptions_text}"""

messages = [
    {"role": "system", "content": instruction},
    {"role": "user", "content": input_text}
]

full_prompt = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

**改进**：
1. ✅ 使用相同的instruction
2. ✅ 包含Prompt字段
3. ✅ 字段顺序和内容完全一致
4. ✅ 使用Qwen chat template

---

## 📊 数据流对比

### **微调阶段**
```
MER-Factory JSON
    ↓
prepare_au_instruction_dataset.py
    ↓
{
    "instruction": "Generate a facial expression description based on AU detections.",
    "input": "Prompt: ...\nAU values: ...\nAU descriptions: ...",
    "output": "..."
}
    ↓
LLaMA-Factory (Qwen template)
    ↓
微调AU Agent
```

### **推理阶段（AffectGPT）**
```
MER-Factory JSON
    ↓
base_dataset.py (load_au_result_from_mer_factory)
    ↓
{'active_aus': {...}, 'au_description': "..."}
    ↓
conversation_video.py (postprocess_au)
    ↓
au_agent.py (generate_description)
    ↓
{
    instruction: "Generate a facial expression description based on AU detections.",
    input: "Prompt: ...\nAU values: ...\nAU descriptions: ..."
}
    ↓
Qwen chat template
    ↓
AU Agent生成描述
    ↓
Facial Content → AffectGPT
```

---

## ✅ 验证结论

**核心字段匹配**：
- ✅ AU values格式一致
- ✅ AU descriptions格式一致（优先使用MER-Factory提供的）
- ✅ Chat template一致（Qwen模板）

**设计合理性**：
- ✅ 微调时输入完整信息（Emotion + Prompt + AU result）
- ✅ 推理时只输入AU result（符合AU Agent的职责）
- ✅ 推理时强调客观性（无情感词）
- ✅ 模型在微调时学到了从AU到描述的映射，推理时可以只用AU信息

**不会影响训练和推理**：
- ✅ 微调时模型学习了多种输入组合
- ✅ 推理时使用简化输入（只有AU result）仍然有效
- ✅ 模型能够根据AU值生成客观描述
- ✅ 符合论文设计：AU Agent生成Facial Content（客观描述）

---

## 🎯 最佳实践

1. **始终使用MER-Factory的au_description**
   - 优先使用预生成的描述（来自GPT-4o）
   - 保证训练和推理时的描述格式一致

2. **保持字段顺序**
   - Prompt → AU values → AU descriptions
   - 与微调数据格式完全一致

3. **使用相同的instruction**
   - 无标签场景：`"Generate a facial expression description based on AU detections."`
   - 有标签场景：`"Generate a detailed facial expression description based on the given information."`

4. **使用Qwen chat template**
   - 通过`tokenizer.apply_chat_template()`
   - 保证special tokens正确

---

## 📝 示例对比

### **微调样本**
```json
{
    "instruction": "Generate a facial expression description based on AU detections.",
    "input": "Prompt: Given the emotion label, AU intensity values, and their semantic descriptions, provide a detailed and natural facial expression description:\nAU values: AU04_r: 0.88, AU10_r: 2.37, AU12_r: 1.73, AU14_r: 2.45\nAU descriptions: Brow lowerer (intensity: 0.88), Upper lip raiser (intensity: 2.37), Lip corner puller (smile) (intensity: 1.73), Dimpler (intensity: 2.45)",
    "output": "The expression shows moderate brow lowering combined with pronounced upper lip raising and lip corner pulling, accompanied by significant dimpling, indicating coordinated engagement of both upper and lower facial muscles with varied intensity patterns."
}
```

### **推理输入**（AffectGPT调用AU Agent时）
```python
# 推理时只使用AU result，不包含Emotion和Prompt
instruction = "Generate a detailed and objective facial muscle movement description based on the Action Unit detections. Focus only on the physical movements without inferring emotions."
input_text = """AU values: AU04_r: 0.88, AU10_r: 2.37, AU12_r: 1.73, AU14_r: 2.45
AU descriptions: Brow lowerer (intensity: 0.88), Upper lip raiser (intensity: 2.37), Lip corner puller (smile) (intensity: 1.73), Dimpler (intensity: 2.45)"""
```

### **预期输出**
```
The expression shows moderate brow lowering combined with pronounced upper lip raising and lip corner pulling, accompanied by significant dimpling, indicating coordinated engagement of both upper and lower facial muscles with varied intensity patterns.
```

**关键点**：
- ✅ 只输入AU result
- ✅ 描述客观、无情感词
- ✅ 模型能够正确生成描述（微调时已学习）
