# 基于 AU Info 的智能8帧采样功能

## 功能概述

本功能实现了基于 MER-Factory 输出的 `au_info` 进行智能视频帧采样，相比传统的均匀采样，能够更好地捕捉视频中的情感峰值信息。

## 采样策略

固定采样 **8帧**，根据峰值帧位置智能选择：

### 策略1：前后都有充足帧 (frames_before >= 2 && frames_after >= 2)
- ✅ 峰值帧 (1帧)
- ✅ 峰值帧前2帧 (peak_index-1, peak_index-2)
- ✅ 峰值帧后2帧 (peak_index+1, peak_index+2)
- ✅ 剩余帧中均匀采样3帧
- **总计：5+3=8帧**

### 策略2：一边只有1帧 (frames_before == 1 || frames_after == 1)
- ✅ 峰值帧 (1帧)
- ✅ 少的一边全取 (1帧)
- ✅ 多的一边取邻近2帧
- ✅ 剩余帧中均匀采样4帧
- **总计：4+4=8帧**

### 策略3：前后都只有1帧 (frames_before == 1 && frames_after == 1)
- ✅ 峰值帧 (1帧)
- ✅ 前1帧 + 后1帧 (2帧)
- ✅ 剩余帧中均匀采样5帧
- **总计：3+5=8帧**

### 策略4：一边为0帧 (frames_before == 0 || frames_after == 0)
- ✅ 峰值帧 (1帧)
- ✅ 非0一边取邻近2帧
- ✅ 剩余帧中均匀采样5帧
- **总计：3+5=8帧**

## 使用方法

### 🎯 支持两种模式

#### 模式A：预提取模式（推荐）

提前提取特征，训练时直接加载。

**方法1：使用Shell脚本**
```bash
cd /home/project/AffectGPT/AffectGPT
bash run_mercaptionplus_extraction.sh
# 选择：选项1 - 智能模式
```

**方法2：直接使用Python**
```bash
python extract_multimodal_features_precompute.py \
    --dataset mercaptionplus \
    --modality frame \
    --frame_sampling emotion_peak \
    --frame_n_frms 8 \
    --video_root /path/to/videos \
    --csv_path /path/to/csv \
    --csv_column name \
    --mer-factory-output /home/project/MER-Factory/output \
    --save_root ./preextracted_features \
    --device cuda:0
```

#### 模式B：实时模式（新增支持）⭐

训练时实时加载视频并智能采样，无需预提取。

**配置训练YAML：**
```yaml
datasets:
  mercaptionplus:
    frame_sampling: "emotion_peak"
    frame_n_frms: 8
    mer_factory_output: "/home/project/MER-Factory/output"  # 关键配置
    use_preextracted_features: False
```

详见：[SMART_SAMPLING_CONFIG.md](./SMART_SAMPLING_CONFIG.md)

## 参数说明

| 参数 | 必需 | 说明 |
|------|------|------|
| `--frame_sampling emotion_peak` | 是 | 启用智能采样模式 |
| `--frame_n_frms 8` | 是 | 固定为8帧 |
| `--mer-factory-output` | 是 | MER-Factory输出目录路径 |

## au_info 文件格式

智能采样需要读取 MER-Factory 生成的 JSON 文件，格式如下：

```json
{
    "au_info": {
        "total_frames": 45,
        "peak_frames": [
            {
                "peak_index": 10,
                "frames_before_peak": 10,
                "frames_after_peak": 34
            }
        ]
    }
}
```

文件路径示例：
```
/home/project/MER-Factory/output/
├── samplenew3_00000070/
│   └── samplenew3_00000070_au_analysis.json
├── samplenew3_00000071/
│   └── samplenew3_00000071_au_analysis.json
...
```

## 测试脚本

运行测试脚本验证采样逻辑：

```bash
python test_smart_sampling.py
```

## 优势

✅ **更好的情感表征**：围绕峰值帧采样，捕捉情感变化关键时刻  
✅ **自适应策略**：根据视频长度自动调整采样策略  
✅ **固定帧数**：保持8帧输出，便于模型训练  
✅ **回退机制**：无au_info时自动回退到均匀采样  

## 输出

采样后的特征保存在：
```
preextracted_features/mercaptionplus/
└── frame_CLIP_VIT_LARGE_emotion_peak_8frms/
    ├── samplenew3_00000070.npy  [8, 768]
    ├── samplenew3_00000071.npy  [8, 768]
    ...
```

## 注意事项

1. **必须先运行 MER-Factory** 生成 au_info
2. **确保路径正确**：`--mer-factory-output` 指向正确的输出目录
3. **CSV文件对应**：CSV中的视频名必须在MER-Factory输出中存在
4. **回退处理**：找不到au_info时自动使用均匀采样

## 与训练配置对接

在训练配置文件中设置：

```yaml
datasets:
  mercaptionplus:
    vis_processor:
      train:
        name: "alpro_video_train"
        image_size: 224
        n_frms: 8  # 与预提取一致
    
use_preextracted_features: True
preextracted_root: "./preextracted_features/mercaptionplus"
frame_sampling: "emotion_peak"  # 指定使用智能采样特征
```

## 性能对比

| 采样方式 | 帧数 | 特点 | 适用场景 |
|----------|------|------|----------|
| uniform | 8 | 均匀分布 | 通用场景 |
| headtail | 6 | 首尾各3帧 | 短视频 |
| **emotion_peak (智能)** | **8** | **峰值+邻近+均匀** | **情感分析（推荐）** |

---

**作者**: AffectGPT Team  
**日期**: 2025-11-11  
**版本**: 1.0
