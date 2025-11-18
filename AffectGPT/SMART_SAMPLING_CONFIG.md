# 智能8帧采样 - 训练配置指南

## 配置方式

智能采样现在支持**两种模式**：

### 模式1：预提取模式（推荐）⭐

预先提取特征，训练时直接加载`.npy`文件。

```yaml
# 训练配置 YAML
datasets:
  mercaptionplus:
    # 数据路径
    vis_root: "/path/to/videos"
    face_root: "/path/to/openface_face"
    wav_root: "/path/to/audio"
    ann_path: "/path/to/annotations.csv"
    
    # Frame采样配置
    frame_n_frms: 8
    frame_sampling: "emotion_peak"  # 使用智能采样
    
    # 预提取特征配置
    use_preextracted_features: True  # ✅ 启用预提取模式
    preextracted_root: "./preextracted_features/mercaptionplus"
    visual_encoder: "CLIP_VIT_LARGE"
    acoustic_encoder: "HUBERT_LARGE"
    
    # 视觉处理器配置
    vis_processor:
      train:
        name: "alpro_video_train"
        image_size: 224
        n_frms: 8
```

**优势：**
- ✅ 训练速度快（无需实时提取）
- ✅ 显存占用低（跳过编码器）
- ✅ 特征可复用

### 模式2：实时模式（新增支持）

训练时实时加载视频并进行智能采样。

```yaml
# 训练配置 YAML
datasets:
  mercaptionplus:
    # 数据路径
    vis_root: "/path/to/videos"
    face_root: "/path/to/openface_face"
    wav_root: "/path/to/audio"
    ann_path: "/path/to/annotations.csv"
    
    # Frame采样配置
    frame_n_frms: 8
    frame_sampling: "emotion_peak"  # 使用智能采样
    
    # ⭐ 新增：MER-Factory输出路径（用于加载au_info）
    mer_factory_output: "/home/project/MER-Factory/output"
    
    # 预提取特征配置
    use_preextracted_features: False  # ❌ 禁用预提取
    
    # 视觉处理器配置
    vis_processor:
      train:
        name: "alpro_video_train"
        image_size: 224
        n_frms: 8
```

**优势：**
- ✅ 无需预提取步骤
- ✅ 支持数据增强（RandomResizedCrop等）
- ✅ 灵活性高

**注意：**
- ⚠️ 需要确保 `mer_factory_output` 路径正确
- ⚠️ 训练速度较慢（需要实时处理）
- ⚠️ 显存占用较高（需要编码器）

## 配置参数说明

| 参数 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `frame_sampling` | string | 是 | 设置为 `"emotion_peak"` 启用智能采样 |
| `frame_n_frms` | int | 是 | 固定为 `8` |
| `mer_factory_output` | string | 实时模式必需 | MER-Factory输出目录路径 |
| `use_preextracted_features` | bool | 否 | `True`=预提取模式，`False`=实时模式 |
| `preextracted_root` | string | 预提取模式必需 | 预提取特征根目录 |

## 完整示例

### 预提取模式完整配置

```yaml
model:
  arch: affectgpt
  model_type: affectgpt
  load_pretrained: True
  pretrained: "/path/to/checkpoint.pth"
  
  # 跳过编码器（使用预提取特征）
  skip_encoders: True

datasets:
  mercaptionplus:
    # 基础路径
    vis_root: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/video"
    face_root: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/openface_face"
    wav_root: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/audio"
    ann_path: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/track2_train_mercaptionplus.csv"
    
    # 采样配置
    frame_n_frms: 8
    frame_sampling: "emotion_peak"
    
    # 预提取配置
    use_preextracted_features: True
    preextracted_root: "./preextracted_features/mercaptionplus"
    visual_encoder: "CLIP_VIT_LARGE"
    acoustic_encoder: "HUBERT_LARGE"
    clips_per_video: 8
    
    # 处理器
    vis_processor:
      train:
        name: "alpro_video_train"
        image_size: 224
        n_frms: 8

run:
  task: video_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 1e-5
  warmup_lr: 1e-6
  batch_size_train: 4
  batch_size_eval: 4
  num_workers: 4
  max_epoch: 10
```

### 实时模式完整配置

```yaml
model:
  arch: affectgpt
  model_type: affectgpt
  load_pretrained: True
  pretrained: "/path/to/checkpoint.pth"
  
  # 不跳过编码器（实时提取特征）
  skip_encoders: False

datasets:
  mercaptionplus:
    # 基础路径
    vis_root: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/video"
    face_root: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/openface_face"
    wav_root: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/audio"
    ann_path: "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/track2_train_mercaptionplus.csv"
    
    # 采样配置
    frame_n_frms: 8
    frame_sampling: "emotion_peak"
    
    # ⭐ MER-Factory输出路径
    mer_factory_output: "/home/project/MER-Factory/output"
    
    # 禁用预提取
    use_preextracted_features: False
    
    # 处理器
    vis_processor:
      train:
        name: "alpro_video_train"
        image_size: 224
        n_frms: 8

run:
  task: video_text_pretrain
  lr_sched: "linear_warmup_cosine_lr"
  init_lr: 1e-4
  min_lr: 1e-5
  warmup_lr: 1e-6
  batch_size_train: 4
  batch_size_eval: 4
  num_workers: 4
  max_epoch: 10
```

## 工作流程

### 预提取模式流程

```
1. 预提取阶段（只需执行一次）
   ↓
   bash run_mercaptionplus_extraction.sh
   选择：选项1 - 智能模式
   ↓
   生成：frame_CLIP_VIT_LARGE_emotion_peak_8frms/*.npy
   
2. 训练阶段
   ↓
   设置 use_preextracted_features: True
   ↓
   训练脚本直接加载 .npy 文件
   ↓
   ✅ 快速训练
```

### 实时模式流程

```
1. 准备阶段
   ↓
   确保 MER-Factory 已生成 au_info
   设置 mer_factory_output 路径
   
2. 训练阶段
   ↓
   设置 use_preextracted_features: False
   设置 mer_factory_output: "/path/to/output"
   ↓
   训练时自动加载视频 → 读取au_info → 智能采样8帧
   ↓
   ✅ 实时训练
```

## 回退机制

如果找不到 `au_info`，系统会自动回退：

**预提取模式：**
- 回退到均匀采样8帧
- 日志：`⚠️ Warning: Failed to load au_info from ...`

**实时模式：**
- 回退到取中间帧（1帧）
- 需要检查视频是否在 MER-Factory 输出中

## 验证配置

训练启动时会打印配置信息：

```
====== Frame Sampling Config ======
Frame frames: 8, Frame sampling: emotion_peak
Face frames: 8, Face sampling: uniform
===================================
```

如果使用实时模式还会显示：
```
[DATASET] Using smart emotion_peak sampling with au_info
```

## 故障排查

### 问题1：实时模式下采样失败

**症状：** 只返回1帧而不是8帧

**原因：** 未设置 `mer_factory_output` 或路径错误

**解决：** 
```yaml
datasets:
  mercaptionplus:
    mer_factory_output: "/home/project/MER-Factory/output"  # 确保路径正确
```

### 问题2：预提取特征文件找不到

**症状：** `frame_feat_path does not exist`

**原因：** 预提取时使用的目录名与配置不匹配

**解决：**
```yaml
# 确保目录名匹配：frame_{visual_encoder}_{frame_sampling}_{frame_n_frms}frms
# 例如：frame_CLIP_VIT_LARGE_emotion_peak_8frms
```

### 问题3：au_info 文件不存在

**症状：** `⚠️ Warning: Failed to load au_info`

**原因：** MER-Factory 未处理该视频

**解决：**
1. 运行 MER-Factory 处理所有视频
2. 确保输出目录结构正确：
   ```
   /home/project/MER-Factory/output/
   ├── video_name1/
   │   └── video_name1_au_analysis.json
   ├── video_name2/
   │   └── video_name2_au_analysis.json
   ```

## 总结

| 特性 | 预提取模式 | 实时模式 |
|------|-----------|----------|
| 训练速度 | ⚡⚡⚡ 很快 | ⚡ 较慢 |
| 显存占用 | 💾 低 | 💾💾 高 |
| 数据增强 | ❌ 不支持 | ✅ 支持 |
| 设置复杂度 | 🔧 需要预提取 | 🔧 直接使用 |
| **推荐场景** | **生产训练** | **实验调试** |

---

**作者**: AffectGPT Team  
**日期**: 2025-11-11  
**版本**: 2.0 (新增实时模式支持)
