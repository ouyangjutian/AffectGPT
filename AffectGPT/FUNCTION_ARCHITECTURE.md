# 特征提取函数架构说明

## 📊 函数调用关系图

```
预提取模式主函数 (extract_dataset_features)
    ↓
    调用: extract_frame_features(video_path, n_frms, sampling, video_name)
    ↓
    ├─ sampling='uniform'  → 标准采样（均匀）
    ├─ sampling='headtail' → 标准采样（头尾）
    └─ sampling='emotion_peak' + video_name 提供
        ↓
        自动转发到: extract_frame_features_smart(video_path, video_name, n_frms=8)
            ↓
            1. 加载 au_info
            2. 计算智能帧索引（4种策略）
            3. 手动加载指定帧
            4. 特征提取
```

## 🎯 函数职责

### 1. `extract_frame_features()` - 统一入口函数 ⭐

**职责**：所有帧特征提取的统一入口

**支持的采样策略**：
- ✅ `uniform` - 均匀采样
- ✅ `headtail` - 头尾采样  
- ✅ `emotion_peak` - 智能采样（自动转发）

**函数签名**：
```python
def extract_frame_features(
    self, 
    video_path: str,
    n_frms: int = 8,
    sampling: str = 'uniform',
    video_name: Optional[str] = None  # emotion_peak模式需要
) -> np.ndarray  # [T, D]
```

**内部逻辑**：
```python
if sampling == 'emotion_peak' and video_name:
    # 转发到智能采样函数
    return self.extract_frame_features_smart(video_path, video_name, n_frms=8)
else:
    # 标准采样（uniform/headtail）
    raw_frame = load_video(video_path, n_frms, sampling=sampling)
    features = self.encoders['visual'](frame, raw_frame)
    return features
```

**使用示例**：
```python
# 均匀采样8帧
features = extractor.extract_frame_features(
    video_path='video.mp4',
    n_frms=8,
    sampling='uniform'
)

# 智能采样8帧（自动转发）
features = extractor.extract_frame_features(
    video_path='video.mp4',
    n_frms=8,
    sampling='emotion_peak',
    video_name='samplenew3_00000070'  # 必需
)
```

---

### 2. `extract_frame_features_smart()` - 智能采样实现 🧠

**职责**：实现基于 au_info 的智能8帧采样

**函数签名**：
```python
def extract_frame_features_smart(
    self,
    video_path: str,
    video_name: str,  # 必需，用于查找au_info
    n_frms: int = 8   # 固定为8
) -> np.ndarray  # [8, D]
```

**内部流程**：
```python
1. 加载 au_info
   au_info = self.load_au_info(video_name)
   
2. 计算智能帧索引
   frame_indices = self.calculate_smart_frame_indices(au_info, total_frames)
   # 返回8个帧索引
   
3. 手动加载指定帧
   raw_frame = self._load_specific_frames(video_path, frame_indices)
   
4. 特征提取
   features = self.encoders['visual'](frame, raw_frame)
   return features  # [8, D]
```

**采样策略**（4种）：
1. **策略1**：前后≥2帧 → 峰值+前2+后2+均匀3
2. **策略2**：一边1帧 → 峰值+1帧+2帧+均匀4
3. **策略3**：前后各1帧 → 峰值+前1+后1+均匀5
4. **策略4**：一边0帧 → 峰值+2帧+均匀5

**使用示例**：
```python
# 直接调用（不推荐，应使用统一入口）
features = extractor.extract_frame_features_smart(
    video_path='video.mp4',
    video_name='samplenew3_00000070'
)
```

---

### 3. 辅助函数

#### `load_au_info(video_name)`
加载MER-Factory的au_info

#### `calculate_smart_frame_indices(au_info, total_frames)`
根据au_info计算8个帧索引

#### `_load_specific_frames(video_path, frame_indices)`
手动加载指定索引的帧

## ✅ 优化后的调用方式

### 主函数中（推荐）✨

```python
# ✅ 推荐：统一使用 extract_frame_features
frame_features = extractor.extract_frame_features(
    video_path=video_path,
    n_frms=args.frame_n_frms,        # 8
    sampling=args.frame_sampling,     # 'emotion_peak'
    video_name=sample_name            # 'samplenew3_00000070'
)

# 它会自动判断：
# - 如果是 emotion_peak + video_name提供 → 调用智能采样
# - 否则 → 调用标准采样
```

### 旧的调用方式（已简化）

```python
# ❌ 旧方式（已移除）：手动判断
if args.frame_sampling == 'emotion_peak':
    frame_features = extractor.extract_frame_features_smart(...)
else:
    frame_features = extractor.extract_frame_features(...)
```

## 📝 设计优势

### 优势1：统一接口
所有采样策略通过同一个函数 `extract_frame_features` 调用，简化使用。

### 优势2：自动路由
根据 `sampling` 参数自动选择合适的实现：
- `uniform/headtail` → 标准实现
- `emotion_peak` → 智能实现

### 优势3：向后兼容
现有代码无需修改，只需添加 `video_name` 参数即可启用智能采样。

### 优势4：清晰职责
- `extract_frame_features` - 对外接口
- `extract_frame_features_smart` - 内部实现

## 🔄 完整调用流程

```
1. 用户调用
   extract_frame_features(
       video_path='video.mp4',
       n_frms=8,
       sampling='emotion_peak',
       video_name='samplenew3_00000070'
   )

2. 函数内部判断
   if sampling == 'emotion_peak' and video_name:
       ↓ 转发到智能采样
   
3. 智能采样处理
   extract_frame_features_smart()
       ↓ 加载 au_info
       ↓ 计算帧索引 [0, 8, 9, 10, 11, 12, 18, 31]
       ↓ 加载指定帧
       ↓ 提取特征
   
4. 返回结果
   features: np.ndarray [8, 768]
```

## 📌 总结

| 函数 | 角色 | 何时使用 | 是否直接调用 |
|------|------|----------|------------|
| `extract_frame_features` | **统一入口** | 所有情况 | ✅ 推荐 |
| `extract_frame_features_smart` | **内部实现** | emotion_peak | ❌ 不推荐 |

**最佳实践**：
```python
# ✅ 总是使用统一入口
features = extractor.extract_frame_features(
    video_path, n_frms, sampling, video_name
)

# ❌ 避免直接调用内部实现
features = extractor.extract_frame_features_smart(...)  # 不推荐
```

---

**作者**: AffectGPT Team  
**日期**: 2025-11-11  
**版本**: 3.0 (统一接口优化)
