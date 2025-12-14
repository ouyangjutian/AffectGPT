# MER-UniBench 快速入门

## 🎯 目标

为9个MER-UniBench数据集预提取Frame的emotion_peak特征，实现**混合模式推理**：
- **Frame**: 预提取emotion_peak（加速600-1200倍）
- **Face**: 实时uniform采样
- **Audio**: 实时处理
- **AU**: 实时CLIP编码

---

## ⚡ 三步走

### **步骤1：预提取Frame特征（一次性，~3.5小时）**

```bash
cd /home/project/AffectGPT/AffectGPT/MER-UniBench

# 运行批量提取
bash run_extract_emotion_peak_batch.sh
```

**输出**：
```
preextracted_features/
├── mer2023/frame_CLIP_VIT_LARGE_emotion_peak_8frms/  (~500MB)
├── mer2024/frame_CLIP_VIT_LARGE_emotion_peak_8frms/  (~600MB)
├── cmumosei/...  (~2.5GB)
├── cmumosi/...   (~2.2GB)
├── iemocap/...   (~5GB)
├── meld/...      (~2.6GB)
├── ovmerdplus/... (~800MB)
├── sims/...      (~2.3GB)
└── simsv2/...    (~2.3GB)
```

---

### **步骤2：配置推理（使用提供的模板）**

使用已创建的配置文件：
```bash
# 示例配置已创建：
ls ../eval_configs/eval_mer2023_frame_preextract.yaml
```

**关键配置**：
```yaml
datasets:
  mer2023:
    # Frame预提取
    frame_sampling: 'emotion_peak'
    use_preextracted_features: True
    preextracted_root: './preextracted_features/mer2023'
    
    # 其他模态实时处理
    use_au_clip_realtime: True
    mer_factory_output: '/home/project/MER-Factory/output'

model:
  skip_encoders: False  # 必须False（实时处理需要编码器）
```

---

### **步骤3：运行推理**

```bash
cd /home/project/AffectGPT/AffectGPT

python inference_hybird.py \
    --cfg-path eval_configs/eval_mer2023_frame_preextract.yaml \
    --dataset mer2023 \
    --ckpt <your_checkpoint_path>
```

**预期输出**：
```
Inference: 100%|██████████| 411/411 [00:45<00:00, 9.12it/s]
                                      ^^^^^^^^^^^^^^^^^^^^
                                      快速！比实时emotion_peak快5倍
```

---

## 📊 性能对比

### **MER2023 (411样本)**

| 方案 | 耗时 | 存储 | 说明 |
|------|------|------|------|
| ❌ 实时emotion_peak | ~4分钟 | 0 | JSON I/O慢 |
| ✅ **Frame预提取（推荐）** | **~45秒** | **500MB** | **本方案** |
| ⚡ 完全预提取 | ~10秒 | 10GB | 极致速度但占用大 |

---

## 🔧 自定义配置

### **只处理部分数据集**

```bash
# 只提取MER2023和MER2024
python3 extract_frame_emotion_peak_batch.py \
    --datasets mer2023 mer2024 \
    --device cuda:0
```

### **修改其他数据集**

复制并修改配置文件：
```bash
cp eval_configs/eval_mer2023_frame_preextract.yaml \
   eval_configs/eval_mer2024_frame_preextract.yaml

# 编辑新文件，修改：
# - datasets.mer2024 (替换mer2023)
# - preextracted_root: './preextracted_features/mer2024'
# - 数据路径
```

---

## ⚠️ 前置要求

### 1. MER-Factory AU分析

emotion_peak采样依赖MER-Factory生成的`au_info`：

```bash
# 检查是否已生成
ls /home/project/MER-Factory/output/mer2023/sample_*/sample_*_au_analysis.json | wc -l

# 如果没有，运行MER-Factory
cd /home/project/MER-Factory
python main.py --dataset mer2023 --modality video
```

### 2. 磁盘空间

确保有足够空间：
- 单个数据集：500MB - 5GB
- 全部9个数据集：~18GB

### 3. GPU内存

推理需要：
- 最小: 8GB（单模态）
- 推荐: 16GB（多模态）
- 最佳: 24GB+（大batch size）

---

## 📁 目录结构

```
/home/project/AffectGPT/AffectGPT/
├── MER-UniBench/                           # 批量处理工具目录
│   ├── extract_frame_emotion_peak_batch.py # 提取脚本
│   ├── run_extract_emotion_peak_batch.sh   # Shell脚本
│   ├── QUICK_START.md                      # 本文件
│   ├── INFERENCE_CONFIG_GUIDE.md           # 配置详解
│   ├── EMOTION_PEAK_PREEXTRACTION_GUIDE.md # 技术文档
│   └── README.md                           # 总览
│
├── preextracted_features/                  # 特征输出目录
│   ├── mer2023/
│   │   └── frame_CLIP_VIT_LARGE_emotion_peak_8frms/
│   ├── mer2024/
│   └── ...
│
├── eval_configs/                           # 推理配置目录
│   ├── eval_mer2023_frame_preextract.yaml  # 示例配置
│   └── ...
│
└── inference_hybird.py                     # 推理脚本
```

---

## ❓ 故障排查

### 问题1: 特征提取失败

```
ValueError: MER-Factory output not found
```

**解决**：
```bash
# 先运行MER-Factory
cd /home/project/MER-Factory
python main.py --dataset mer2023 --modality video
```

---

### 问题2: 推理还是很慢

```
# 推理用时: ~4分钟（应该~45秒）
```

**检查**：
```bash
# 1. 特征是否存在？
ls preextracted_features/mer2023/frame_CLIP_VIT_LARGE_emotion_peak_8frms/*.npy | wc -l
# 应该显示411

# 2. 配置是否正确？
grep -A3 "use_preextracted_features" eval_configs/eval_mer2023_frame_preextract.yaml
# 应该显示: use_preextracted_features: True

# 3. 路径是否正确？
grep "preextracted_root" eval_configs/eval_mer2023_frame_preextract.yaml
# 应该是相对或绝对正确路径
```

---

### 问题3: 编码器错误

```
RuntimeError: Visual encoder is None
```

**解决**：
```yaml
# 确保配置中：
model:
  skip_encoders: False  # ← 必须是False！
```

---

## 📚 相关文档

- `README.md`: 项目总览和设计理念
- `INFERENCE_CONFIG_GUIDE.md`: 详细配置说明
- `EMOTION_PEAK_PREEXTRACTION_GUIDE.md`: 技术原理和性能分析

---

## 🎉 成功标志

运行推理时看到：
```
====== Inference Frame Sampling Config ======
Frame frames: 8, Frame sampling: emotion_peak
Face frames: 8, Face sampling: uniform

✅ Frame features loaded (preextracted)
⏳ Face: real-time processing
⏳ Audio: real-time processing
⏳ AU: real-time CLIP encoding

Inference: 100%|██████████| 411/411 [00:45<00:00, 9.12it/s]
```

恭喜！你已成功配置混合模式推理 🎊
