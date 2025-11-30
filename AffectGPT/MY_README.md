# AffectGPT 项目完整指南

> **最后更新**: 2024-11-23 21:30  
> **说明**: 本文档整合了所有功能模块的使用说明和配置指南  
> **版本**: v2.4 - 集成训练可视化，自动保存曲线图

---

## 📚 目录

1. [项目概述](#项目概述)
2. [AU处理三种模式（重要）](#au处理三种模式重要)
3. [快速开始 - 不使用AU Agent](#快速开始---不使用au-agent)
4. [训练可视化（自动保存曲线图）](#训练可视化自动保存曲线图)
5. [Pre-Fusion内部机制详解](#pre-fusion内部机制详解)
6. [训练采样机制详解](#训练采样机制详解)
7. [AU Agent 集成指南](#au-agent-集成指南)
8. [AU特征提取与训练](#au特征提取与训练)
9. [编码器跳过策略](#编码器跳过策略)
10. [Frame采样策略](#frame采样策略)
11. [预提取特征优化](#预提取特征优化)
12. [配置文件对比](#配置文件对比)
13. [常见问题](#常见问题)

---

## 项目概述

AffectGPT是一个多模态情感识别系统，集成了视频、音频、文本和AU（Action Unit）等多种模态。

### 核心功能

- **多模态融合**: Frame, Face, Audio, Text, AU
- **AU Agent**: 从AU值生成自然语言描述
- **智能采样**: 基于情感峰值的帧选择策略
- **预提取特征**: 减少训练时的计算开销

### 系统架构

```
视频/音频输入
    ↓
OpenFace AU分析
    ↓
AU Agent生成描述
    ↓
多模态特征提取
    ↓
AffectGPT推理
    ↓
情感识别结果
```

---

## AU处理三种模式（重要）

> **核心思想**: AU Agent只在生成JSON时使用一次，训练和推理直接使用CLIP编码生成的描述

### 📊 三种模式对比

| 模式 | AU Agent | CLIP加载 | 显存占用 | 速度 | 适用场景 |
|------|---------|---------|---------|------|---------|
| **模式1: 预提取特征** | ❌ 不使用 | ❌ 不需要 | 🟢 15GB | ⚡ 最快 | ✅ 训练（推荐） |
| **模式2: 实时CLIP编码** | ❌ 不使用 | ✅ CLIP ViT-B/32 | 🟡 17GB | 🚀 较快 | ✅ 推理（推荐） |
| **模式3: AU Agent** | ✅ 使用 | ✅ CLIP + AU Agent | 🔴 30GB | 🐌 慢 | ⚠️ 不推荐 |

### 推荐方案 ⭐

**训练**: 预提取特征模式  
**推理**: 实时CLIP编码模式

```yaml
# 训练配置
model:
  use_au_agent: False  # ❌ 不使用AU Agent
  skip_encoders: True  # ✅ 跳过编码器

datasets:
  mercaptionplus:
    use_preextracted_features: True  # ✅ 使用预提取特征
    preextracted_root: './preextracted_features/mercaptionplus'

# 推理配置
model:
  use_au_agent: False  # ❌ 不使用AU Agent
  skip_encoders: False  # ❌ 不跳过（需要实时编码）

inference:
  use_au_clip_realtime: True  # ✅ 实时CLIP编码
  mer_factory_output: '/home/project/MER-Factory/output'
```

**优点**:
- ✅ 训练最快（预提取）
- ✅ 推理灵活（实时编码）
- ✅ 显存占用小（15-17GB vs 30GB）
- ✅ 不需要AU Agent模型

---

## 快速开始 - 不使用AU Agent

### 🎯 完整工作流程

```
步骤1: 生成AU描述（使用AU Agent，只运行一次）
MER-Factory → AU Agent → JSON文件（含summary_description）

步骤2: 提取训练特征（只运行一次）
JSON → CLIP编码 → .npy文件

步骤3: 训练（不使用AU Agent）
.npy文件 → 直接加载 → AffectGPT训练

步骤4: 推理（不使用AU Agent）
JSON → summary_description → CLIP实时编码 → AffectGPT推理
```

### 步骤1: 生成AU描述

```bash
cd /home/project/MER-Factory

# 批量处理所有数据集
python batch_extract_au_multi_datasets.py \
    --mode 2 \              # 测试集模式
    --gen-method 1 \        # AU Agent生成
    --datasets 1            # 全部10个数据集

# 或后台运行
nohup python batch_extract_au_multi_datasets.py --mode 2 --gen-method 1 --datasets 1 > batch.log 2>&1 &
```

### 步骤2: 提取CLIP特征（训练用）

```bash
cd /home/project/AffectGPT/AffectGPT

# 使用提供的脚本
bash extract_mercaptionplus_features.sh

# 或手动运行
python extract_multimodal_features_precompute.py \
    --dataset mercaptionplus \
    --modality au \
    --device cuda:0 \
    --mer-factory-output /home/project/MER-Factory/output/MERCaptionPlus \
    --csv_path /path/to/train.csv \
    --csv_column name \
    --save_root ./preextracted_features
```

### 步骤3: 训练（不使用AU Agent）

```bash
cd /home/project/AffectGPT/AffectGPT

# 使用推荐配置文件
python train.py \
    --cfg-path train_configs/recommended_train_with_preextracted_au.yaml
```

**配置要点**:
```yaml
model:
  use_au_agent: False  # ❌ 不使用AU Agent
  skip_encoders: True  # ✅ 跳过编码器加载

datasets:
  mercaptionplus:
    use_preextracted_features: True  # ✅ 使用预提取特征
    preextracted_root: './preextracted_features/mercaptionplus'
```

### 步骤4: 推理（不使用AU Agent）

```bash
# 使用推荐配置文件
python inference.py \
    --cfg-path eval_configs/recommended_inference_with_clip_realtime.yaml
```

**配置要点**:
```yaml
model:
  use_au_agent: False  # ❌ 不使用AU Agent
  skip_encoders: False  # ❌ 推理不跳过

inference:
  use_au_clip_realtime: True  # ✅ 实时CLIP编码
  mer_factory_output: '/home/project/MER-Factory/output'
  # 注意：路径会自动补充数据集名称，无需手动指定
  # 实际读取路径: {mer_factory_output}/{dataset}/{video_name}/{video_name}_au_analysis.json
  # 例如: /home/project/MER-Factory/output/MER2023/sample_00000905/sample_00000905_au_analysis.json
```

### ⚠️ 路径配置重要说明

**MER-Factory输出路径结构**:
```
/home/project/MER-Factory/output/
├── MER2023/              # 数据集名称（自动从dataset类获取）
│   └── sample_00000905/  # 视频名称
│       └── sample_00000905_au_analysis.json
├── MER2024/
│   └── sample_xxx/
│       └── sample_xxx_au_analysis.json
└── MERCaptionPlus/
    └── samplenew3_00000120/
        └── samplenew3_00000120_au_analysis.json
```

**配置方式**:
```yaml
# ✅ 正确：只配置根路径
inference:
  mer_factory_output: '/home/project/MER-Factory/output'

# ❌ 错误：不要手动添加数据集名称
inference:
  mer_factory_output: '/home/project/MER-Factory/output/MER2023'  # 会导致路径重复
```

**自动路径构建**:
- 推理时，代码会自动从数据集类获取 `self.dataset` 属性（如 `'MER2023'`）
- 自动构建完整路径: `{mer_factory_output}/{dataset}/{video_name}/{video_name}_au_analysis.json`
- 支持多个数据集推理，每个数据集使用相同的 `mer_factory_output` 根路径

### 📊 性能优势

| 指标 | 新方案 | 旧方案（AU Agent） | 提升 |
|------|--------|-------------------|------|
| **显存占用** | 15-17GB | 30GB | 节省43% |
| **训练速度** | 基准 | 慢3倍 | 提升3倍 |
| **推理速度** | 基准 | 慢3倍 | 提升3倍 |
| **GPU要求** | 1x 20GB | 2x 20GB | 节省1张卡 |

---

## AU Agent 集成指南

### ✅ AU Agent功能

- ✅ 从AU值生成客观的肌肉运动描述（无情感词）
- ✅ 支持从MER-Factory JSON加载AU result
- ✅ 训练和推理统一使用AU Agent
- ✅ 显存优化：支持单独GPU运行AU Agent

### 📊 完整数据流

#### **训练阶段**
```
1. MER-Factory生成AU result (OpenFace only)
   └── {sample_name}_au_analysis.json

2. base_dataset.py加载
   └── _load_au_result_from_mer_factory()
   └── 返回: {'active_aus': {...}, 'au_description': "..."}

3. conversation_video.py处理
   └── postprocess_au() 使用AU Agent
   └── AU Agent生成Facial Content描述
       输入: AU values + AU descriptions (只有AU result)
       输出: 客观的肌肉运动描述（无情感词）
   └── 转换为text tokens → 输入AffectGPT训练
```

#### **推理阶段**
```
1. MER-Factory生成AU result (同训练)
2. base_dataset.py加载 (同训练)
3. conversation_video.py处理 (同训练)
   └── AU Agent生成描述 → AffectGPT推理 → 情感识别结果
```

### 🚀 使用步骤

#### 步骤1: 生成AU分析结果

```bash
cd /home/project/MER-Factory

# 使用批处理脚本（推荐）
python batch_extract_au_multi_datasets.py \
    --mode 2 \              # 测试集模式
    --gen-method 1 \        # AU Agent生成
    --datasets 1            # 全部数据集

# 后台运行
nohup python batch_extract_au_multi_datasets.py > batch_run.log 2>&1 &
```

**输出目录结构**:
```
/home/project/MER-Factory/output/
├── MERCaptionPlus/
│   ├── samplenew3_00000120/
│   │   └── samplenew3_00000120_au_analysis.json
├── MER2023/
├── MER2024/
└── ... (其他8个数据集)
```

**JSON文件结构**:
```json
{
  "per_frame_au_descriptions": [
    {
      "frame": 1,
      "timestamp": 0.0,
      "au_description": "Upper lip raiser (intensity: 1.06), ...",
      "active_aus": {"AU10_r": 1.06, "AU12_r": 1.14, ...},
      "is_peak_frame": false,
      "fine_grained_description": "system\n...\nuser\n...\nassistant\n完整描述"
    }
  ],
  "au_info": {
    "total_frames": 44,
    "peak_frames": [{"peak_index": 42, ...}]
  },
  "summary_description": {
    "1": "纯净描述（仅assistant部分，用于CLIP特征提取）",
    "35": "The expression features moderate brow lowering...",
    "69": "The facial expression demonstrates..."
  }
}
```

#### 步骤2: 配置训练/推理

**训练配置** (`train_configs/*.yaml`):
```yaml
model:
  arch: affectgpt
  
  # AU Agent配置
  use_au_agent: True  # 训练时使用AU Agent
  au_agent_base_model: "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
  au_agent_lora_weights: "/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora"
  au_agent_device: "cuda:1"  # AU Agent独占GPU 1

datasets:
  mercaptionplus:
    # MER-Factory输出路径
    mer_factory_output: '/home/project/MER-Factory/output'
    
    # Frame采样配置
    frame_n_frms: 8
    frame_sampling: 'uniform'  # 或 'emotion_peak'
```

**推理配置** (`eval_configs/*.yaml`):
```yaml
model:
  use_au_agent: True  # 推理时也使用AU Agent
  au_agent_base_model: "..."
  au_agent_lora_weights: "..."

datasets:
  mer2023:  # 或其他数据集
    mer_factory_output: '/home/project/MER-Factory/output'
```

#### 步骤3: 训练/推理

```bash
# 训练
python train.py --cfg-path train_configs/xxx.yaml

# 推理
python inference.py --cfg-path eval_configs/xxx.yaml
```

### ⚠️ 显存要求

| 组件 | 显存需求 | GPU分配 |
|------|---------|---------|
| AffectGPT (7B) | ~15GB | cuda:0 |
| AU Agent (7B + LoRA) | ~15GB | cuda:1 |
| **总计** | **~30GB** | 2x GPU |

**推荐配置**:
- 单卡训练: 80GB A100
- 双卡训练: 2x 40GB A100 (AffectGPT在GPU 0, AU Agent在GPU 1)
- DDP训练: 使用GPU 0,2,3训练，AU Agent在GPU 1

---

## AU特征提取与训练

### 🔄 更新说明 (2024-11-23)

- **JSON字段**: `summary_description`
- **描述内容**: `summary_description` 只包含纯净的assistant部分
- **用途**: 专门用于CLIP特征提取

### 📁 目录结构

#### MER-Factory输出
```
/home/project/MER-Factory/output/
├── MERCaptionPlus/
│   ├── samplenew3_00000120/
│   │   └── samplenew3_00000120_au_analysis.json
└── ... (其他9个数据集)
```

#### AffectGPT预提取特征
```
/home/project/AffectGPT/AffectGPT/preextracted_features/
└── mercaptionplus/
    └── au_CLIP_VITB32_8frms/
        ├── samplenew3_00000120.npy  # [N, 512]
        └── ...
```

### 🚀 特征提取流程

#### 步骤1: 测试单样本

```bash
cd /home/project/AffectGPT/AffectGPT

# 测试单个样本的完整流程
python test_single_sample.py --sample samplenew3_00000120
```

#### 步骤2: 批量提取CLIP特征

```bash
# 方式1: 使用提供的脚本（推荐）
bash extract_mercaptionplus_features.sh

# 方式2: 手动指定参数
python extract_multimodal_features_precompute.py \
    --dataset mercaptionplus \
    --modality au \
    --device cuda:0 \
    --mer-factory-output /home/project/MER-Factory/output/MERCaptionPlus \
    --csv_path /path/to/train_file.csv \
    --csv_column name \
    --save_root ./preextracted_features
```

**参数说明**:

| 参数 | 说明 | 示例值 |
|------|------|--------|
| `--dataset` | 数据集名称 | `mercaptionplus` |
| `--modality` | 提取模态 | `au` (仅AU特征) |
| `--device` | GPU设备 | `cuda:0` |
| `--mer-factory-output` | MER-Factory输出目录 | `/home/project/MER-Factory/output/MERCaptionPlus` |
| `--csv_path` | 样本列表CSV | 包含name列的CSV文件 |
| `--save_root` | 保存目录 | `./preextracted_features` |

#### 步骤3: 验证数据完整性

```bash
# 运行验证脚本
python verify_au_pipeline.py
```

**验证内容**:
1. ✅ MER-Factory JSON文件存在
2. ✅ 包含 `summary_description` 字段
3. ✅ CLIP特征文件存在且格式正确
4. ✅ 描述数量与特征维度匹配

#### 步骤4: 训练使用预提取特征

**修改训练配置**:
```yaml
datasets:
  mercaptionplus:
    # 启用预提取特征模式
    use_preextracted_features: True
    preextracted_root: './preextracted_features/mercaptionplus'
    
    # 编码器配置（用于构建特征路径）
    visual_encoder: 'CLIP_VIT_LARGE'
    acoustic_encoder: 'HUBERT_LARGE'
    clips_per_video: 8
```

**运行训练**:
```bash
python train.py --cfg-path train_configs/your_config.yaml
```

### 📊 数据加载逻辑

`base_dataset.py` 加载优先级:

1. **预提取模式** (`use_preextracted_features=True`):
   ```python
   # 从 .npy 文件加载CLIP特征
   au_feat_path = preextracted_root/au_CLIP_VITB32_8frms/sample_name.npy
   au = torch.from_numpy(np.load(au_feat_path))  # [N, 512]
   ```

2. **AU Agent模式** (`use_preextracted_features=False`):
   ```python
   # 从 MER-Factory JSON实时加载，使用AU Agent推理
   au = self._load_au_result_from_mer_factory(video_name)
   ```

### 🎯 性能对比

| 模式 | 训练速度 | 显存占用 | 特征一致性 |
|------|---------|---------|-----------|
| 预提取特征 | ⚡ 快 | 💾 低 | ✅ 一致 |
| 实时生成 | 🐌 慢 | 🔥 高 | ⚠️ 可能变化 |

**建议**:
- ✅ 训练使用预提取特征（速度快、显存低）
- ⏸️ 推理可实时生成（更灵活）

---

## Frame采样策略

### 📋 采样策略对比

#### 1. Uniform采样（均匀采样）
```yaml
frame_n_frms: 8
frame_sampling: 'uniform'
```

**特点**:
- 从视频中均匀采样8帧
- 覆盖整个视频时长
- 适合表情变化平缓的视频

#### 2. Emotion Peak采样（情感峰值采样）
```yaml
frame_n_frms: 8
frame_sampling: 'emotion_peak'
```

**特点**:
- 基于AU峰值帧智能选择
- 选择表情变化最明显的区域
- 需要MER-Factory的 `au_info` 数据

**采样逻辑**:
```python
def calculate_smart_frame_indices(au_info, n_frms=8):
    """智能选择关键帧"""
    peak_frames = au_info.get('peak_frames', [])
    
    if not peak_frames:
        # 无峰值，回退到均匀采样
        return uniform_sample(total_frames, n_frms)
    
    # 1. 峰值帧必选
    selected_indices.add(peak_index)
    
    # 2. 峰值前后帧（如 ±3帧）
    for offset in range(-3, 4):
        if 0 <= peak_index + offset < total_frames:
            selected_indices.add(peak_index + offset)
    
    # 3. 补充均匀采样的帧
    while len(selected_indices) < n_frms:
        # 从未选择的帧中均匀采样
        ...
    
    return sorted(selected_indices)[:n_frms]
```

### 🔧 配置示例

#### 训练配置
```yaml
datasets:
  mercaptionplus:
    # Frame采样
    frame_n_frms: 8
    frame_sampling: 'uniform'  # 训练建议uniform
    
    # MER-Factory路径（emotion_peak需要）
    mer_factory_output: '/home/project/MER-Factory/output'
```

#### 推理配置
```yaml
datasets:
  mer2023:
    # Frame采样
    frame_n_frms: 8
    frame_sampling: 'emotion_peak'  # 推理可用emotion_peak
    
    # MER-Factory路径
    mer_factory_output: '/home/project/MER-Factory/output'
```

### ⚠️ 注意事项

1. **emotion_peak需要au_info**
   - 确保MER-Factory已生成 `au_info` 字段
   - 如果缺失，自动回退到uniform采样

2. **采样数量**
   - 推荐 `frame_n_frms: 8` (默认)
   - 峰值模式可能选择少于8帧（如果峰值区域较小）

3. **兼容性**
   - 所有采样策略都兼容预提取特征模式
   - AU模态独立于Frame采样策略

---

## 编码器跳过策略

### 📊 `skip_encoders` 参数详解

`skip_encoders` 控制**模型初始化时**是否加载编码器：
- CLIP ViT-Large（Frame/Face特征）
- HuBERT-Large（Audio特征）

**注意**: 不影响AU特征编码（AU用CLIP ViT-B/32或预提取.npy）

### 🔄 配置策略对比

| 策略 | 训练skip_encoders | 推理skip_encoders | 适用场景 |
|------|------------------|------------------|---------|
| **策略1: 原始** | False | False | 开发测试 |
| **策略2: 推荐** ⭐ | True | False | 生产环境 |
| **策略3: 极致** | True | True | 固定数据集 |

### 推荐策略 ⭐

**训练配置**:
```yaml
model:
  skip_encoders: True  # ✅ 跳过（配合预提取特征）
  
datasets:
  mercaptionplus:
    use_preextracted_features: True
```

**推理配置**:
```yaml
model:
  skip_encoders: False  # ❌ 不跳过（保持灵活性）
  
inference:
  use_au_clip_realtime: True  # AU实时CLIP编码
```

### 影响分析

#### 训练时 `skip_encoders: True`
- ✅ 节省3GB显存
- ✅ 必须使用预提取特征
- ✅ 训练速度最快

#### 推理时 `skip_encoders: False`
- ✅ 可实时编码新数据
- ✅ 灵活性高
- ⚠️ 显存稍大（17-18GB）

---

## 配置文件对比

### 📋 AU配置项清单

| 配置项 | 说明 | 推荐值 |
|--------|------|--------|
| `preextracted_au_dim` | AU特征维度 | 512 |
| `frozen_au_Qformer` | AU Q-Former冻结 | False |
| `frozen_au_proj` | AU投影层冻结 | False |
| `au_fusion_type` | AU融合方式 | attention |
| `num_au_query_token` | AU query token数 | 1 |

### 配置文件对比

#### 原始配置 (`emercoarse_highlevelfilter4_outputhybird_bestsetup_bestfusion_lz_face_frame_au.yaml`)

```yaml
model:
  skip_encoders: False
  use_au_agent: True  # ✅ 使用AU Agent
  
datasets:
  mercaptionplus:
    use_preextracted_features: False
```

#### 新训练配置 (`recommended_train_with_preextracted_au.yaml`) ⭐

```yaml
model:
  skip_encoders: True  # ✅ 训练时跳过
  use_au_agent: False  # ❌ 不使用AU Agent
  
datasets:
  mercaptionplus:
    use_preextracted_features: True  # ✅ 预提取
```

#### 新推理配置 (`recommended_inference_with_clip_realtime.yaml`) ⭐

```yaml
model:
  skip_encoders: False  # ❌ 推理不跳过
  use_au_agent: False  # ❌ 不使用AU Agent
  
inference:
  use_au_clip_realtime: True  # ✅ 实时CLIP编码
```

### 性能对比

| 配置 | 显存 | 训练速度 | 推理速度 | 灵活性 |
|------|------|---------|---------|--------|
| 原始配置 | 30GB | 基准 | 基准 | 高 |
| 新训练配置 | 15GB | 快3倍 | - | 低 |
| 新推理配置 | 17GB | - | 快3倍 | 高 |

---

## 预提取特征优化

### 🎯 优化目标

- ⚡ **加速训练**: 避免每个epoch重复编码
- 💾 **节省显存**: 不需要加载CLIP/HuBERT编码器
- ✅ **一致性**: 所有epoch使用相同特征

### 📊 支持的模态

| 模态 | 编码器 | 输出维度 | 特征目录 |
|------|--------|---------|----------|
| Frame | CLIP ViT-L | [8, 768] | `frame_CLIP_VIT_LARGE_8frms` |
| Face | CLIP ViT-L | [8, 768] | `face_CLIP_VIT_LARGE_8frms` |
| Audio | HuBERT-Large | [8, 1024] | `audio_HUBERT_LARGE_8clips` |
| AU | CLIP ViT-B/32 | [N, 512] | `au_CLIP_VITB32_8frms` |

### 🚀 提取所有模态特征

```bash
cd /home/project/AffectGPT/AffectGPT

# 提取所有模态（Frame, Face, Audio, AU）
python extract_multimodal_features_precompute.py \
    --dataset mercaptionplus \
    --modality all \
    --device cuda:0 \
    --video_root /path/to/videos \
    --face_root /path/to/faces \
    --audio_root /path/to/audios \
    --mer-factory-output /home/project/MER-Factory/output/MERCaptionPlus \
    --csv_path /path/to/train.csv \
    --save_root ./preextracted_features \
    --visual_encoder CLIP_VIT_LARGE \
    --acoustic_encoder HUBERT_LARGE

# 仅提取特定模态
python extract_multimodal_features_precompute.py \
    --dataset mercaptionplus \
    --modality frame \  # 或 face, audio, au
    ...
```

### 📁 预提取特征目录结构

```
preextracted_features/
└── mercaptionplus/
    ├── frame_CLIP_VIT_LARGE_8frms/
    │   ├── sample_00000120.npy  # [8, 768]
    │   └── ...
    ├── face_CLIP_VIT_LARGE_8frms/
    │   ├── sample_00000120.npy  # [8, 768]
    │   └── ...
    ├── audio_HUBERT_LARGE_8clips/
    │   ├── sample_00000120.npy  # [8, 1024]
    │   └── ...
    └── au_CLIP_VITB32_8frms/
        ├── sample_00000120.npy  # [N, 512]
        └── ...
```

### 🔧 训练配置

**启用预提取特征**:
```yaml
model:
  # 完全跳过编码器加载（节省显存）
  skip_encoders: True
  preextracted_visual_dim: 768
  preextracted_acoustic_dim: 1024

datasets:
  mercaptionplus:
    # 启用预提取特征
    use_preextracted_features: True
    preextracted_root: './preextracted_features/mercaptionplus'
    
    # 编码器配置（用于构建路径）
    visual_encoder: 'CLIP_VIT_LARGE'
    acoustic_encoder: 'HUBERT_LARGE'
    clips_per_video: 8
```

### 📊 性能提升

| 指标 | 实时编码 | 预提取特征 | 提升 |
|------|---------|-----------|------|
| 训练速度 | 100% | 150%+ | ⚡ +50% |
| 显存占用 | 20GB | 15GB | 💾 -25% |
| 特征一致性 | 可能变化 | 完全一致 | ✅ 100% |

---

## 训练可视化（自动保存曲线图）

### 📊 功能说明

训练过程中**自动生成**学习率和Loss曲线图，无需额外操作！

### ✨ 特点

- ✅ **自动保存**: 每个epoch结束自动保存图片
- ✅ **无需额外脚本**: 集成在训练代码中
- ✅ **高质量图表**: 标准版(150 DPI) + 高清版(300 DPI)
- ✅ **多维度展示**: 学习率、Loss、Epoch统计
- ✅ **一键开关**: 配置文件控制启用/禁用

### 🎨 生成的图表

每个epoch结束时自动生成包含4个子图的曲线图：

1. **学习率 vs 步数（线性）**: 查看warmup和衰减过程
2. **学习率 vs 步数（对数）**: 更清晰地看到学习率变化
3. **Loss vs 步数**: 原始loss + 平滑曲线（100步窗口）
4. **Loss vs Epoch**: 每个epoch的平均loss + 标准差

### 📁 输出位置

```
output/your_experiment/training_curves/
├── training_curves_epoch1.png       # Epoch 1结束时的曲线
├── training_curves_epoch2.png       # Epoch 2结束时的曲线
├── ...
├── training_curves_epoch10.png      # Epoch 10结束时的曲线
├── training_curves_hd_epoch5.png    # 高清版（每5个epoch）
├── training_curves_hd_epoch10.png   # 高清版（每5个epoch）
└── training_data_epoch10.npz        # 原始数据（可选）
```

### ⚙️ 配置方法

#### 默认启用（推荐）

训练可视化**默认启用**，无需任何配置：

```bash
# 直接训练即可
python train.py --cfg-path train_configs/your_config.yaml
```

#### 手动控制

在配置文件中添加（可选）：

```yaml
run_cfg:
  # ... 其他配置 ...
  
  visualize_training: True    # 启用训练可视化（默认True）
  # visualize_training: False # 禁用训练可视化
```

### 📊 实时查看统计

每个epoch结束时，会在训练日志中自动打印统计信息：

```
======================================================================
📊 Training Statistics
======================================================================
  Total Steps:          31,327
  Current Epoch:        1
  Current Learning Rate: 9.82e-05
  Latest Loss:          0.654321
  Recent 100 Avg Loss:  0.723456
  Best Loss:            0.345678 (Step 28934)
  Max Learning Rate:    1.00e-04
  Min Learning Rate:    1.23e-06
======================================================================
```

### 🖼️ 查看图片

#### 方式1: 本地训练

```bash
# 查看生成的图片
ls -lh output/your_experiment/training_curves/

# 直接打开查看（Linux桌面环境）
xdg-open output/your_experiment/training_curves/training_curves_epoch10.png
```

#### 方式2: 远程服务器训练

```bash
# 方式A: 使用scp下载到本地
scp user@server:/path/to/output/your_experiment/training_curves/*.png ./local_folder/

# 方式B: 使用rsync同步
rsync -avz user@server:/path/to/output/your_experiment/training_curves/ ./local_folder/

# 方式C: 使用VS Code Remote
# 直接在VS Code中浏览和查看图片
```

### 💡 使用技巧

#### 1. 实时监控进度

```bash
# 每隔一段时间下载最新图片
while true; do
    scp user@server:/path/to/training_curves/training_curves_epoch*.png ./
    sleep 300  # 每5分钟同步一次
done
```

#### 2. 训练中期检查

```python
# 如果需要中途查看曲线，可以手动调用
# 在训练代码中添加（可选）
if epoch % 5 == 0:  # 每5个epoch
    visualizer.plot_and_save(suffix=f'_epoch{epoch}_checkpoint')
```

#### 3. 对比不同实验

```bash
# 将不同实验的曲线放在一起对比
experiment1/training_curves/training_curves_epoch10.png
experiment2/training_curves/training_curves_epoch10.png
```

### 🎯 实际示例

#### 预期的学习率曲线

```
Learning Rate (log scale)
    │
1e-4│     ╱──────────╲
    │    ╱            ╲___
    │   ╱                 ╲___
1e-6│  ╱                      ╲___
    └───────────────────────────────> Steps
    0  6265        156635       313270
    ↑ Warmup      ↑ Peak         ↑ End
```

#### 预期的Loss曲线

```
Loss
  4│╲
   │ ╲___
  2│     ╲___
   │         ╲___
  0│             ────────
   └────────────────────────> Steps
   快速下降    平稳     收敛
```

### 🔧 高级功能

#### 保存原始数据

可视化器会自动保存原始数据为`.npz`文件：

```python
# 加载数据进行自定义分析
import numpy as np

data = np.load('output/your_experiment/training_curves/training_data_epoch10.npz')
steps = data['steps']
lrs = data['lrs']
losses = data['losses']
epochs = data['epochs']

# 自定义绘图
import matplotlib.pyplot as plt
plt.plot(steps, losses)
plt.savefig('custom_plot.png')
```

#### 禁用高清版保存

如果不需要高清版（节省空间），可以修改代码：

```python
# 在 training_visualizer.py 中
# 注释掉高清版保存的代码（第121-180行）
```

### ⚠️ 注意事项

1. **存储空间**: 每个epoch约生成1-2MB图片，10个epoch约10-20MB
2. **仅主进程**: 多GPU训练时，只有主进程（rank 0）生成图片
3. **matplotlib后端**: 使用Agg后端，无需GUI环境
4. **自动覆盖**: 同名文件会被覆盖（建议定期备份重要曲线）

### ❓ 常见问题

#### Q1: 没有生成图片？

**A**: 检查以下几点：
```bash
# 1. 确认输出目录
ls output/your_experiment/

# 2. 检查训练日志
grep "Training curves saved" train.log

# 3. 确认可视化已启用
grep "visualize_training" train_configs/your_config.yaml
```

#### Q2: 图片不清晰？

**A**: 使用高清版：
```bash
# 查找高清版图片（300 DPI）
ls output/your_experiment/training_curves/*_hd.png
```

#### Q3: 能否修改图表样式？

**A**: 可以编辑 `my_affectgpt/common/training_visualizer.py`：
```python
# 修改颜色、线宽、图表大小等
plt.style.use('seaborn-v0_8-darkgrid')  # 第72行
fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # 第75行
```

#### Q4: 能否在训练结束后重新生成？

**A**: 可以，数据已保存：
```python
# 加载数据重新绘制
from my_affectgpt.common.training_visualizer import TrainingVisualizer

vis = TrainingVisualizer('output/new_plots', enabled=True)
vis.load_data('output/your_experiment/training_curves/training_data_epoch10.npz')
vis.plot_and_save(suffix='_final')
```

### 📝 完整工作流程

```bash
# 1. 启动训练（可视化自动启用）
python train.py --cfg-path train_configs/your_config.yaml > train.log 2>&1 &

# 2. 查看训练日志
tail -f train.log

# 3. 每个epoch结束时，日志会显示：
#    ✅ Training curves saved: output/.../training_curves_epoch1.png
#    📊 Training Statistics
#    ... (统计信息) ...

# 4. 下载图片到本地查看（远程训练）
scp user@server:/path/to/output/*/training_curves/*.png ./

# 5. 查看曲线，监控训练进度
# - 学习率是否正常衰减？
# - Loss是否平稳下降？
# - 是否过拟合或欠拟合？

# 6. 训练结束后，所有epoch的曲线都已保存
ls -lh output/your_experiment/training_curves/
```

### 🎓 总结

**核心优势**：
- ✅ 零配置：默认启用，无需额外操作
- ✅ 自动化：每个epoch自动保存，无需手动触发
- ✅ 高质量：专业的图表样式和统计信息
- ✅ 轻量级：集成在训练流程，无性能影响

**使用建议**：
- 📊 定期查看曲线，及时发现训练问题
- 💾 重要实验建议备份曲线图
- 🔍 对比不同实验的曲线，选择最佳配置

---

## Pre-Fusion内部机制详解

### 🎯 概述

Pre-Fusion是AffectGPT中**Audio和Face/Video模态融合**的核心机制，使用**Cross-Attention (Q-Former)**实现跨模态信息整合。

### 📊 核心架构

```
输入: Audio特征 + Face特征
  ↓
特征对齐（Linear投影到统一维度）
  ↓
特征拼接（Concat）+ 位置编码
  ↓
Cross-Attention (Q-Former)
  ├─ Q (Query): 16个可学习的查询向量
  ├─ K (Key): 来自Audio+Face的concat特征
  └─ V (Value): 来自Audio+Face的concat特征
  ↓
输出: 16个融合token
  ↓
投影到LLM空间
```

### 🔍 Q, K, V 详解

#### Query (Q) - "我想要什么信息"

- **来源**: `self.multi_query_tokens` - 可学习参数
- **维度**: `[batch, 16, 768]`
- **特性**: 
  - ✅ 固定数量（16个）
  - ✅ 训练过程中学习最优query策略
  - ✅ 每个query关注不同的跨模态信息方面

**示例**:
```python
Q1: 关注整体情感强度
Q2: 关注面部表情细节
Q3: 关注音视频一致性
...
Q16: 关注全局上下文
```

#### Key & Value (K, V) - "这里有什么信息"

- **来源**: Audio特征 + Face特征的concat
- **维度**: `[batch, 40, 1024]` (假设8个audio帧 + 32个face帧)
- **处理流程**:
  ```python
  Za = Linear(Audio)        # [batch, 8, 1024]
  Zf = Linear(Face)         # [batch, 32, 1024]
  Z_concat = Concat(Za, Zf) # [batch, 40, 1024]
  Z_kv = Z_concat + PosEmb  # 添加位置编码
  ```

#### Cross-Attention计算

```python
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

Q: [batch, 16, 768]   # 16个queries
K: [batch, 40, 768]   # 40个时间步的keys
V: [batch, 40, 768]   # 40个时间步的values

Attention_Scores: [batch, 16, 40]  # 每个query对40个时间步的关注度
Output: [batch, 16, 768]           # 16个融合后的token
```

### 📐 维度变化示例

```
输入:
  Audio: [3, 8, 1024]
  Face:  [3, 32, 1024]

对齐:
  Audio': [3, 8, 1024]
  Face':  [3, 32, 1024]

拼接:
  Concat: [3, 40, 1024]  # 8+32=40

位置编码:
  Z_kv: [3, 40, 1024]

Query准备:
  Zq: [3, 16, 768]

Cross-Attention:
  Output: [3, 16, 768]

投影到LLM:
  Ef': [3, 16, 4096]  # 投影到LLaMA空间
```

### 💡 设计优势

| 特性 | 说明 |
|------|------|
| **维度压缩** | 40个时间步 → 16个token |
| **跨模态融合** | Audio + Face信息有效整合 |
| **自适应学习** | Query学习最优的信息提取策略 |
| **固定输出** | 输出维度固定，便于后续LLM处理 |

### 🔗 详细文档

- **技术详解**: 参见 `PRE_FUSION_MECHANISM.md`
  - 完整代码分析
  - 详细计算流程
  - 参数配置说明
  
- **可视化图表**: 参见 `PRE_FUSION_VISUAL.md`
  - 数据流图
  - Q-K-V交互示意
  - 维度变化全流程
  - 图书馆查询类比

### 📊 Attention权重示例

```
Query 1 (关注情感强度):
  Audio帧:  [0.05, 0.08, 0.15, 0.20, 0.10, 0.05, 0.03, 0.02]
  Face帧:   [0.01, 0.01, ..., 0.05, 0.08, 0.10, 0.04, ...]
  → 主要关注音频的中间段和面部的后段

Query 2 (关注表情细节):
  Audio帧:  [0.01, 0.01, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01]
  Face帧:   [0.12, 0.10, 0.09, ..., 0.05, 0.04, 0.03, ...]
  → 主要关注面部的前几帧
```

### 🎓 关键代码

**文件**: `my_affectgpt/models/affectgpt.py`

**函数**: `encode_multi_qformer` (Line 843-876)

```python
def encode_multi_qformer(self, video_hidden_state, audio_hidden_state):
    # 特征对齐
    video_hidden_state = self.multi_video_embs(video_hidden_state)
    audio_hidden_state = self.multi_audio_embs(audio_hidden_state)
    
    # 拼接
    multi_hidden_state = torch.concat([video_hidden_state, audio_hidden_state], axis=1)
    
    # 位置编码
    multi_hidden_state = multi_hidden_state + multi_position_embeddings
    
    # Cross-Attention (Q-Former)
    multi_query_tokens = self.multi_query_tokens.expand(batch, -1, -1)
    multi_query_output = self.multi_Qformer.bert(
        query_embeds=multi_query_tokens,      # Q
        encoder_hidden_states=multi_hidden_state,  # K, V
        encoder_attention_mask=frame_atts,
        return_dict=True,
    )
    
    # 投影到LLM
    inputs_llama = self.multi_llama_proj(multi_query_output.last_hidden_state)
    return multi_hidden, inputs_llama
```

---

## 训练采样机制详解

### 🎯 核心概念

训练不是简单地 "每个epoch随机抽1000个样本"，而是使用**迭代器（Iterator）模式**：

#### 配置参数
```yaml
run_cfg:
  max_epoch: 30           # 总epoch数
  iters_per_epoch: 1000   # 每个epoch的迭代次数
  warmup_steps: 1000      # warmup步数
```

#### 训练日志解析
```
training sample number: 5000         # 数据集总样本数
Loaded 5000 records for train split  # 加载的训练样本数
Start training epoch 1, 1000 iters per inner epoch  # 每个epoch跑1000次迭代
```

### 📊 采样原理

#### 1. DataLoader 配置
```python
# runner_base.py
sampler = DistributedSampler(
    dataset,
    shuffle=True,  # ✅ 训练时随机打乱
    num_replicas=world_size,
    rank=rank
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    sampler=sampler,
    shuffle=False,  # sampler已处理打乱，这里不需要
    drop_last=True  # 丢弃最后不完整的batch
)
```

#### 2. 迭代器模式
```python
# base_task.py - train_epoch()
data_loader = iter(data_loader)  # 转换为迭代器

for i in range(iters_per_epoch):  # 循环1000次
    samples = next(data_loader)   # 每次取1个batch
    # ... 前向传播、反向传播、更新参数
```

### 🔄 实际训练流程

假设配置如下：
- 总样本数: 5000
- batch_size: 8
- iters_per_epoch: 1000
- max_epoch: 30

#### Epoch 1
1. **打乱数据**: DistributedSampler 随机打乱5000个样本的顺序
2. **迭代训练**: 
   - iter 0: 取batch [样本0-7]
   - iter 1: 取batch [样本8-15]
   - ...
   - iter 624: 取batch [样本4992-4999] → 5000个样本用完
   - iter 625: **循环回到开头**，取batch [样本0-7]
   - ...
   - iter 999: 训练完成，进入下一个epoch

**关键**: 迭代器会**循环使用**数据集！

#### Epoch 2
1. **重新打乱**: DistributedSampler 重新随机打乱（顺序与epoch 1不同）
2. **继续迭代**: 同样跑1000次迭代

### 📈 计算关系

```python
# 每个epoch实际访问的样本数量
actual_samples_per_epoch = iters_per_epoch * batch_size
                         = 1000 * 8 = 8000 个样本

# 数据集循环次数
dataset_cycles = actual_samples_per_epoch / total_samples
               = 8000 / 5000 = 1.6 轮

# 总训练步数
total_steps = max_epoch * iters_per_epoch
            = 30 * 1000 = 30000 步

# 每个样本被访问次数（平均）
samples_seen_per_data = (total_steps * batch_size) / total_samples
                      = (30000 * 8) / 5000 = 48 次
```

### ⚠️ 重要特性

#### 1. 循环采样
- **不是**: 每个epoch只用5000个样本中的1000个
- **而是**: 每个epoch跑1000次迭代，会**循环使用**数据集
- 5000个样本用完后，**自动回到开头继续**（每个epoch重新打乱）

#### 2. 随机性保证
```python
# DistributedSampler 每个epoch都会重新打乱
def set_epoch(self, epoch):
    self.epoch = epoch
    # 使用epoch作为随机种子，保证每个epoch顺序不同
```

#### 3. 为什么用迭代器模式？

| 对比 | 迭代器模式 | 传统epoch模式 |
|------|-----------|--------------|
| **灵活性** | 高（精确控制步数） | 低（必须跑完整个数据集） |
| **小数据集** | 可循环利用 | 每个epoch很短 |
| **学习率调度** | 基于步数（精确） | 基于epoch（粗糙） |
| **分布式训练** | 各GPU步数一致 | 可能不一致 |

### 💡 实际示例

#### 日志示例
```
Train: data epoch: [1]  [  0/1000]  eta: 0:38:28  lr: 0.00001000  loss: 8.37608719
Train: data epoch: [1]  [ 50/1000]  eta: 0:04:58  lr: 0.00005000  loss: 2.65583491
Train: data epoch: [1]  [100/1000]  eta: 0:04:58  lr: 0.00010000  loss: 1.85704851
...
Train: data epoch: [1]  [150/1000]  eta: 0:04:24  lr: 0.00001000  loss: 1.63498379
```

- `[100/1000]`: 当前epoch的第100次迭代（共1000次）
- 每次迭代处理 batch_size 个样本
- 1000次迭代后进入下一个epoch

### 🎓 总结

**核心要点**:
1. ✅ 每个epoch固定跑 **1000次迭代**（不是1000个样本）
2. ✅ 每次迭代取 **batch_size个样本**（如8个）
3. ✅ 数据集会**循环使用**（5000样本跑1.6轮）
4. ✅ 每个epoch **重新随机打乱**顺序
5. ✅ 学习率基于**总步数**调度（30000步）

**优势**:
- 精确控制训练步数
- 适合小数据集（充分利用）
- 学习率调度更平滑
- 分布式训练更稳定

### 🔥 Warmup 机制详解

#### 什么是 Warmup？

**Warmup** 是学习率预热机制，在训练初期**逐步增加**学习率，避免初始梯度过大导致模型不稳定。

#### 配置参数
```yaml
run_cfg:
  warmup_steps: 1000      # warmup步数
  warmup_lr: 1e-6         # 起始学习率（可选，默认用init_lr）
  init_lr: 1e-4           # 目标学习率（warmup结束时）
  min_lr: 0               # 最小学习率（cosine衰减结束时）
```

#### 学习率调度策略

**LinearWarmupCosineLR**: Warmup线性增长 + Cosine衰减

```python
total_cur_step = cur_epoch * iters_per_epoch + cur_step

if total_cur_step < warmup_steps:
    # 阶段1: Warmup (0 → 1000步)
    lr = warmup_start_lr + (init_lr - warmup_start_lr) * (total_cur_step / warmup_steps)
    # 线性增长: 1e-6 → 1e-4
else:
    # 阶段2: Cosine衰减 (1000 → 30000步)
    progress = (total_cur_step - warmup_steps) / (total_steps - warmup_steps)
    lr = min_lr + (init_lr - min_lr) * 0.5 * (1 + cos(π * progress))
    # 余弦衰减: 1e-4 → 0
```

#### 实际示例

假设配置：
- `warmup_steps: 1000`
- `warmup_lr: 1e-6` (起始)
- `init_lr: 1e-4` (warmup结束)
- `min_lr: 0` (训练结束)
- `max_epoch: 30`
- `iters_per_epoch: 1000`
- 总步数: 30,000

**学习率变化曲线**:
```
Step 0:       lr = 1e-6     (warmup开始)
Step 500:     lr = 5e-5     (warmup中期，线性增长)
Step 1000:    lr = 1e-4     (warmup结束，达到峰值) ← warmup_steps
Step 10000:   lr ≈ 8e-5     (cosine衰减)
Step 20000:   lr ≈ 5e-5     (继续衰减)
Step 30000:   lr → 0        (衰减到最小值)
```

#### 可视化

```
Learning Rate
    │
1e-4│     ╱──────────╲
    │    ╱            ╲
    │   ╱              ╲
    │  ╱                ╲___
1e-6│ ╱                      ╲___
    │╱                            ╲___
  0 └────────────────────────────────────> Steps
    0   1000              15000          30000
       ↑warmup            ↑中期           ↑结束
```

#### 训练日志对应

您的日志中可以看到学习率变化：
```
Train: [  0/1000]  lr: 0.00001000  # Step 0, warmup开始
Train: [ 50/1000]  lr: 0.00005000  # Step 50, warmup中
Train: [100/1000]  lr: 0.00010000  # Step 100, warmup中
Train: [150/1000]  lr: 0.00001000  # Step 150, 已进入cosine衰减期
```

**注意**: 您的日志显示第150步lr=0.00001000，说明**warmup已完成**，正在cosine衰减。

#### 为什么需要 Warmup？

| 问题 | 不用Warmup | 用Warmup |
|------|-----------|---------|
| **初始梯度** | 可能很大 | 逐步适应 |
| **参数更新** | 剧烈震荡 | 平稳过渡 |
| **训练稳定性** | 容易崩溃 | 更加稳定 |
| **收敛速度** | 可能变慢 | 更快收敛 |
| **最终效果** | 可能较差 | 性能更好 |

#### 最佳实践

```yaml
# 推荐配置
warmup_steps: 1000              # 约为总步数的3-5%
warmup_lr: 1e-6                 # 约为init_lr的1/100
init_lr: 1e-4                   # 根据batch_size调整
min_lr: 0                       # 或设为init_lr的1/100

# 计算公式
warmup_steps = total_steps * 0.03  # 总步数的3%
warmup_lr = init_lr / 100          # 初始学习率的1%
```

#### 调试技巧

**查看学习率曲线**:
```python
# 训练日志中提取lr
import re
with open('train.log') as f:
    lrs = re.findall(r'lr: (\d+\.\d+)', f.read())
    lrs = [float(lr) for lr in lrs]
    
# 绘制曲线
import matplotlib.pyplot as plt
plt.plot(lrs)
plt.xlabel('Step')
plt.ylabel('Learning Rate')
plt.title('LR Schedule')
plt.show()
```

### 💡 Warmup 常见问题

#### Q1: warmup_steps 设多少合适？
**A**: 一般为总步数的 **3-5%**
```python
total_steps = max_epoch * iters_per_epoch = 30 * 1000 = 30,000
warmup_steps = total_steps * 0.03 ≈ 1000  ✅
```

#### Q2: warmup太短或太长会怎样？
- **太短** (如100步): 学习率增长太快，可能不稳定
- **太长** (如5000步): 浪费训练时间，收敛变慢
- **合适** (1000步): 平衡稳定性和效率

#### Q3: 为什么日志显示lr在150步就很小？
**A**: 可能原因：
1. `warmup_steps < 150`（warmup已结束）
2. Cosine衰减已经开始
3. 检查配置文件确认 `warmup_steps` 值

#### Q4: 能否不用warmup？
**A**: 可以，但**不推荐**：
- 大模型微调：**必须**用warmup
- 小数据集：建议用warmup
- 从头训练：强烈建议用warmup

---

## 快速开始

### 🎬 完整工作流程

#### 1. 环境准备

```bash
# 激活conda环境
conda activate vllm2

# 检查GPU
nvidia-smi
```

#### 2. 生成AU分析（MER-Factory）

```bash
cd /home/project/MER-Factory

# 批量处理所有数据集
python batch_extract_au_multi_datasets.py

# 或后台运行
nohup python batch_extract_au_multi_datasets.py > batch_run.log 2>&1 &
```

#### 3. 提取CLIP特征（AffectGPT）

```bash
cd /home/project/AffectGPT/AffectGPT

# 测试单样本
python test_single_sample.py

# 批量提取
bash extract_mercaptionplus_features.sh

# 验证
python verify_au_pipeline.py
```

#### 4. 训练模型

```bash
# 修改配置文件
# vim train_configs/your_config.yaml

# 运行训练
python train.py --cfg-path train_configs/your_config.yaml
```

#### 5. 推理测试

```bash
# 修改推理配置
# vim eval_configs/your_config.yaml

# 运行推理
python inference.py --cfg-path eval_configs/your_config.yaml
```

### 📝 配置文件模板

**训练配置模板**:
```yaml
model:
  arch: affectgpt
  
  # 预提取特征优化
  skip_encoders: True
  use_preextracted_features: True
  
  # AU Agent配置
  use_au_agent: True
  au_agent_base_model: "/path/to/Qwen2.5-7B-Instruct"
  au_agent_lora_weights: "/path/to/au_agent_lora"
  au_agent_device: "cuda:1"

datasets:
  mercaptionplus:
    data_type: video
    face_or_frame: 'multiface_audio_face_frame_text'
    
    # Frame采样
    frame_n_frms: 8
    frame_sampling: 'uniform'
    
    # 预提取特征
    use_preextracted_features: True
    preextracted_root: './preextracted_features/mercaptionplus'
    
    # MER-Factory输出
    mer_factory_output: '/home/project/MER-Factory/output'
```

---

## 常见问题

### ❓ AU Agent相关

#### Q1: AU Agent生成失败
```
⚠️ AU Agent生成失败: xxx
```

**原因**:
- AU Agent模型未正确加载
- 显存不足
- 配置路径错误

**解决**:
1. 检查 `use_au_agent: True`
2. 检查AU Agent模型路径
3. 检查GPU显存（需要~15GB）
4. 确认AU Agent在单独的GPU上

#### Q2: AU result加载失败
```
⚠️ AU result加载失败: sample_xxx
```

**原因**:
- MER-Factory JSON文件不存在
- JSON文件路径配置错误

**解决**:
1. 检查 `mer_factory_output` 路径
2. 确认JSON文件存在: `{mer_factory_output}/{sample_name}/{sample_name}_au_analysis.json`
3. 重新运行MER-Factory批处理

### ❓ 特征提取相关

#### Q3: summary_description为空
```
⚠️ summary_description不存在或为空
```

**原因**:
- MER-Factory未使用AU Agent生成描述
- JSON文件是旧格式

**解决**:
```bash
# 重新运行，确保使用AU Agent模式
cd /home/project/MER-Factory
python batch_extract_au_multi_datasets.py --mode 2 --gen-method 1
```

#### Q4: CLIP特征维度不匹配
```
❌ Expected shape (N, 512), got (N, 768)
```

**原因**:
- 使用了错误的CLIP模型
- 特征文件版本不兼容

**解决**:
1. AU特征应使用CLIP ViT-B/32（输出512维）
2. 删除旧的特征文件重新提取
3. 确认 `extract_multimodal_features_precompute.py` 使用正确的模型

### ❓ 训练相关

#### Q5: 显存不足
```
CUDA out of memory
```

**解决**:
1. **启用预提取特征**: `use_preextracted_features: True`
2. **跳过编码器加载**: `skip_encoders: True`
3. **减小batch size**: `batch_size: 1`
4. **使用梯度累积**: `gradient_accumulation_steps: 4`
5. **AU Agent单独GPU**: `au_agent_device: "cuda:1"`

#### Q6: 数据加载慢
```
DataLoader is too slow
```

**解决**:
1. 使用预提取特征（最有效）
2. 增加 `num_workers`
3. 使用SSD存储特征文件
4. 启用数据预加载

### ❓ 采样策略相关

#### Q7: emotion_peak采样失败
```
⚠️ No peak frames found, falling back to uniform sampling
```

**原因**:
- MER-Factory JSON缺少 `au_info` 字段
- 视频AU强度过低

**解决**:
1. 自动回退到uniform采样（正常行为）
2. 如需峰值采样，重新运行MER-Factory确保生成 `au_info`

---

## 📞 技术支持

### 文件清单

**脚本文件**:
- `batch_extract_au_multi_datasets.py` - MER-Factory批处理
- `extract_multimodal_features_precompute.py` - 特征提取
- `extract_mercaptionplus_features.sh` - 快捷提取脚本
- `test_single_sample.py` - 单样本测试
- `verify_au_pipeline.py` - 流程验证

**配置文件**:
- `train_configs/*.yaml` - 训练配置
- `eval_configs/*.yaml` - 推理配置

**文档文件**:
- `MY_README.md` - 本文档（主文档）

### 相关资源

- **MER-Factory**: `/home/project/MER-Factory/`
- **AffectGPT**: `/home/project/AffectGPT/AffectGPT/`
- **AU Agent模型**: `/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora/`
- **预提取特征**: `./preextracted_features/`

---

## 📈 版本历史

### v2.5.2 (2024-11-24 19:25)
- ✅ **推理Reasoning输出**: 推理默认输出推理过程，不只是分类结果
- ✅ **可控输出模式**: 添加`--no_reasoning`参数控制是否输出reasoning
- ✅ **Prompt优化**: 使用"Please infer the person's emotional state and provide your reasoning process."

### v2.5.1 (2024-11-24 16:20)
- ✅ **dtype不匹配修复**: 修复CLIP编码Float32与模型Half不匹配的问题
- ✅ **日志输出优化**: CLIP模型加载全局只输出一次，避免刷屏
- ✅ **警告信息精简**: 文件缺失/加载失败只提示前几次
- ✅ **性能优化**: CLIP特征直接在GPU上转换为half，减少CPU-GPU传输

### v2.5 (2024-11-24 16:10)
- ✅ **AU推理模式修复**: 修复推理时AU数据类型错误的问题
- ✅ **CLIP实时编码**: 推理时自动启用`use_au_clip_realtime`模式
- ✅ **详细调试信息**: 添加AU加载和CLIP编码的详细日志输出
- ✅ **安全检查增强**: 在conversation层添加AU数据类型检查，避免崩溃
- ✅ **错误提示优化**: 提供清晰的错误信息和解决建议

### v2.4 (2024-11-23 21:30)
- ✅ **训练可视化集成**: 自动生成学习率和Loss曲线图，无需额外脚本
- ✅ **自动保存图片**: 每个epoch结束自动保存标准版和高清版曲线图
- ✅ **统计信息输出**: 实时显示训练统计（步数、学习率、最佳loss等）
- ✅ **零配置使用**: 默认启用，可通过配置文件一键开关
- ✅ **数据持久化**: 自动保存原始数据为.npz文件，支持后续分析

### v2.3 (2024-11-23 20:40)
- ✅ **训练采样机制详解**: 添加完整的迭代器模式训练机制说明
- ✅ **循环采样原理**: 详细解释数据集如何循环使用和随机打乱
- ✅ **Warmup机制详解**: 完整的学习率预热机制说明，包括公式、可视化和最佳实践
- ✅ **学习率调度**: LinearWarmupCosineLR策略详解（线性增长+余弦衰减）
- ✅ **计算关系说明**: 样本访问次数、总步数、warmup比例等详细计算
- ✅ **实际示例**: 基于5000样本、1000 iters_per_epoch的完整示例
- ✅ **调试技巧**: 学习率曲线提取和可视化方法

### v2.2 (2024-11-23 16:45)
- ✅ **修复路径bug**: 修复推理时AU JSON路径缺少数据集名称层级的问题
- ✅ **自动路径构建**: 代码现在自动从 `self.dataset` 获取数据集名称并构建完整路径
- ✅ **多数据集支持**: 推理时可以用同一个 `mer_factory_output` 根路径处理多个数据集
- ✅ **路径说明文档**: 在MY_README.md中添加详细的路径配置说明

### v2.1 (2024-11-23 16:30)
- ✅ **整合所有MD文档**: 删除零散MD文件，所有内容统一在 `MY_README.md`
- ✅ **AU三种模式**: 添加模式对比（预提取/实时CLIP/AU Agent）
- ✅ **快速开始指南**: 不使用AU Agent的完整流程
- ✅ **编码器跳过策略**: 详细说明 `skip_encoders` 配置
- ✅ **配置文件对比**: 原始配置 vs 新配置完整对比
- ✅ **新增配置文件**: `recommended_train_with_preextracted_au.yaml` 和 `recommended_inference_with_clip_realtime.yaml`
- ✅ **代码修改**: `base_dataset.py` 新增 `_load_au_clip_features_from_json()` 方法

### v2.0 (2024-11-23)
- ✅ 整合所有MD文档到 `MY_README.md`
- ✅ 更新AU特征提取流程（使用 `summary_description`）
- ✅ 添加完整的验证和测试工具
- ✅ 优化文档结构和使用说明

### v2024-11-22
- ✅ 集成AU Agent到训练和推理流程
- ✅ 支持Frame采样策略（uniform/emotion_peak）
- ✅ 添加预提取特征优化

### v2024-11-21
- ✅ 实现AU Agent LoRA微调
- ✅ 集成MER-Factory批处理
- ✅ 初始项目架构

---

## 📄 许可证

本项目遵循原AffectGPT项目许可证。

---

**最后更新**: 2024-11-23  
**维护者**: Project Team  
**联系方式**: 请通过GitHub Issues反馈问题
