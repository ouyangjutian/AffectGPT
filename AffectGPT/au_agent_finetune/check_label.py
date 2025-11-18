#!/usr/bin/env python3
"""检查MER-Factory输出中是否有情感标签"""

import json

json_path = '/home/project/MER-Factory/output/samplenew3_00043253/samplenew3_00043253_au_analysis.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print("="*60)
print("Checking for Emotion Labels in MER-Factory Output")
print("="*60)

print("\n所有字段:")
for key in data.keys():
    print(f"  - {key}")

# 检查是否有情感相关字段
emotion_fields = ['emotion', 'label', 'emotion_label', 'sentiment', 'chronological_emotion_peaks']

print("\n情感相关字段:")
for field in emotion_fields:
    if field in data:
        print(f"  ✅ {field}: {data[field]}")
    else:
        print(f"  ❌ {field}: 不存在")

# 如果有chronological_emotion_peaks，查看详细内容
if 'chronological_emotion_peaks' in data:
    print(f"\nchronological_emotion_peaks 内容:")
    print(f"  {data['chronological_emotion_peaks']}")

# 查看per_frame_au_descriptions的一个示例
if 'per_frame_au_descriptions' in data and data['per_frame_au_descriptions']:
    print(f"\nper_frame_au_descriptions 第一帧:")
    first_frame = data['per_frame_au_descriptions'][0]
    for key, value in first_frame.items():
        if key != 'active_aus':
            print(f"  {key}: {value}")

print("\n" + "="*60)
