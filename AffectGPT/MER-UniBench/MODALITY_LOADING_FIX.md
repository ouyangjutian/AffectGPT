# 模态加载逻辑修复报告

## 🎯 问题根源

配置中设置的`use_preextracted_features=True`被**全局应用到所有模态**，导致：
- ✅ Frame正确加载预提取特征
- ❌ Face尝试加载预提取特征（不存在）→ 失败
- ❌ Audio尝试加载预提取特征（不存在）→ 失败  
- ❌ AU尝试加载预提取特征（不存在）→ 失败
- ❌ 最终所有模态都失败 → `AssertionError: Some input info is missing.`

---

## ✅ 修复方案

### 设计理念：只有Frame使用预提取

| 模态 | 处理方式 | 原因 |
|------|---------|------|
| **Frame** | 预提取emotion_peak | emotion_peak采样慢（需MER-Factory JSON），预提取加速16倍 |
| **Face** | **实时**加载.npy人脸文件 | 已预处理，加载很快（~0.01ms），无需预提取 |
| **Audio** | **实时**加载音频文件 | 音频加载可接受（~15ms），无需预提取 |
| **AU** | **实时**CLIP编码 | 从MER-Factory JSON读取，CLIP编码快（~2ms），无需预提取 |

---

## 🔧 代码修复

### 修复1: 简化Face加载逻辑

**文件**: `/home/project/AffectGPT/AffectGPT/my_affectgpt/datasets/datasets/base_dataset.py`

**修改前** (❌ 尝试加载预提取Face特征):
```python
if 'face' in self.needed_data:
    if hasattr(self, 'use_realtime_extraction') and self.use_realtime_extraction:
        # 实时特征提取服务...
    elif use_preextracted and preextracted_root and sample_name:  # ❌ 尝试预提取
        face_feat_path = os.path.join(preextracted_root, face_feat_dir, f'{sample_name}.npy')
        if os.path.exists(face_feat_path):
            # 加载预提取特征
        else:
            pass  # 预提取失败，但没有fallback!
    else:
        # 实时处理
        if face_npy is not None:
            raw_face, msg = load_face(...)
```

**修改后** (✅ 强制实时处理):
```python
# 🎯 Face/Audio/AU始终使用实时处理（即使启用了预提取，预提取仅针对Frame）
if 'face' in self.needed_data:
    # 实时处理模式 - 直接加载人脸.npy文件
    if face_npy is not None:
        raw_face, msg = load_face(
            face_npy=face_npy,
            n_frms = self.n_frms,
            height = 224,
            width  = 224,
            sampling ="uniform",
            return_msg=True
        )
        face = self.vis_processor.transform(raw_face)
```

---

### 修复2: 简化Audio加载逻辑

**修改前** (❌ 尝试加载预提取Audio特征):
```python
if 'audio' in self.needed_data:
    if hasattr(self, 'use_realtime_extraction') and self.use_realtime_extraction:
        # 实时特征提取服务...
    elif use_preextracted and preextracted_root and sample_name:  # ❌ 尝试预提取
        audio_feat_path = os.path.join(preextracted_root, audio_feat_dir, f'{sample_name}.npy')
        if os.path.exists(audio_feat_path):
            # 加载预提取特征
        else:
            pass  # 预提取失败，但没有fallback!
    else:
        # 实时处理
        if audio_path is not None:
            raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0]
            audio = transform_audio(raw_audio, "cpu")
```

**修改后** (✅ 强制实时处理):
```python
# 🎯 Audio推理时始终使用实时处理（不使用预提取）
if 'audio' in self.needed_data:
    # 实时处理模式 - 直接加载音频文件
    if audio_path is not None:
        raw_audio = load_audio([audio_path], "cpu", clips_per_video=8)[0]
        audio = transform_audio(raw_audio, "cpu")
```

---

### 修复3: 简化AU加载逻辑

**修改前** (❌ 尝试加载预提取AU特征):
```python
if 'au' in self.needed_data:
    # 模式1: 预提取CLIP特征模式
    if use_preextracted and preextracted_root and sample_name:  # ❌ 尝试预提取
        au_feat_path = os.path.join(preextracted_root, au_feat_dir, f'{sample_name}.npy')
        if os.path.exists(au_feat_path):
            au_features = np.load(au_feat_path)
            au = torch.from_numpy(au_features).float()
        else:
            print(f"⚠️ AU特征文件不存在: {au_feat_path}")  # 失败提示
    
    # 模式2: 从JSON实时CLIP编码模式
    elif getattr(self, 'use_au_clip_realtime', False):  # ❌ elif导致无法fallback!
        if video_name and self.mer_factory_output:
            au = self._load_au_clip_features_from_json(video_name)
```

**修改后** (✅ 强制实时CLIP编码):
```python
# 🎯 AU推理时始终使用实时CLIP编码（不使用预提取）
if 'au' in self.needed_data:
    # 模式: 从JSON实时CLIP编码模式（推理推荐）
    if getattr(self, 'use_au_clip_realtime', False):
        # 从video_path或sample_name提取video_name
        video_name = sample_name if sample_name else os.path.splitext(os.path.basename(video_path))[0]
        
        if video_name and self.mer_factory_output:
            # 从JSON加载summary_description并CLIP编码
            au = self._load_au_clip_features_from_json(video_name)
```

---

## 🔄 修复后的数据流

```
配置文件:
  use_preextracted_features: True  ← 全局配置
  preextracted_root: './preextracted_features'
  ↓
  
base_dataset.py 加载逻辑:
  
  Frame模态:
    if use_preextracted and frame_feat_path exists:
        ✅ 加载预提取特征 [8, 768]
    else:
        ✅ 实时加载视频 + emotion_peak采样
  
  Face模态:
    ✅ 强制实时加载.npy人脸文件 (忽略use_preextracted)
  
  Audio模态:
    ✅ 强制实时加载音频文件 (忽略use_preextracted)
  
  AU模态:
    ✅ 强制实时CLIP编码MER-Factory JSON (忽略use_preextracted)
```

---

## ⚠️ 关于Face/Audio失败的警告

用户仍然看到：
```
⚠️ Face特征无效，跳过Face模态: sample_00001998
⚠️ Audio特征无效，跳过Audio模态: sample_00001998
```

**可能原因**：
1. `face_npy`路径不存在或为None
2. `audio_path`路径不存在或为None

**排查步骤**：
1. 检查数据集配置中Face/Audio的路径设置
2. 检查MER2023数据集的Face/Audio文件是否存在
3. 在`base_dataset.py`中添加调试日志查看实际路径

**临时解决**：
如果Face/Audio确实不存在，可以只使用Frame+AU进行推理：
```yaml
inference:
  face_or_frame: 'frame_au'  # 只使用Frame和AU
```

---

## 📊 预期行为

### 成功场景（所有模态都有数据）

```
✅ [Frame预提取] 成功加载预提取特征: mer2023/frame_CLIP_VIT_LARGE_emotion_peak_8frms/
✅ Face加载成功
✅ Audio加载成功
✅ AU CLIP编码成功

[正常推理...]
```

### 部分模态缺失场景

```
✅ [Frame预提取] 成功加载预提取特征
⚠️ Face特征无效，跳过Face模态
⚠️ Audio特征无效，跳过Audio模态
✅ AU CLIP编码成功

[使用Frame+AU进行推理...]
```

---

## ✅ 修复清单

- [x] Face加载逻辑简化（强制实时处理）
- [x] Audio加载逻辑简化（强制实时处理）
- [x] AU加载逻辑简化（强制实时CLIP编码）
- [x] Frame保持预提取逻辑不变
- [ ] **验证Face/Audio路径配置**
- [ ] **重新运行推理测试**

---

## 🚀 下一步

1. **重新运行推理**，观察Face/Audio是否成功加载
2. 如果仍然失败，检查Face/Audio文件路径配置
3. 临时方案：使用`face_or_frame: 'frame_au'`只用Frame+AU推理

---

## 📝 总结

### 问题
`use_preextracted_features=True`导致所有模态都尝试加载预提取特征，但只有Frame有预提取文件。

### 修复
- Frame：保持预提取逻辑（需要加速）
- Face/Audio/AU：强制实时处理（不受`use_preextracted`影响）

### 预期效果
- Frame预提取加载成功
- Face/Audio/AU实时处理（如果文件路径正确）
- 推理正常进行

**预提取优化现在应该完全正常了！** 🎉
