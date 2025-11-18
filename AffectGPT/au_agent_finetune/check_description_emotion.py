#!/usr/bin/env python3
"""检查fine_grained_descriptions是否包含情感词汇"""

import json
import re

json_path = '/home/project/MER-Factory/output/samplenew3_00043253/samplenew3_00043253_au_analysis.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print("="*60)
print("检查生成的描述是否包含情感词汇")
print("="*60)

# 常见情感词汇列表
emotion_keywords = [
    'anger', 'angry', 'happiness', 'happy', 'sadness', 'sad', 
    'fear', 'fearful', 'disgust', 'disgusted', 'surprise', 'surprised',
    'joy', 'joyful', 'frustration', 'frustrated', 'concern', 'concerned'
]

if 'fine_grained_descriptions' in data:
    descriptions = data['fine_grained_descriptions']
    
    print(f"\n✅ Found {len(descriptions)} frame descriptions")
    
    for frame_idx, desc in list(descriptions.items())[:5]:
        print(f"\n--- Frame {frame_idx} ---")
        print(f"Description: {desc}")
        
        # 检查是否包含情感词汇
        desc_lower = desc.lower()
        found_emotions = [kw for kw in emotion_keywords if kw in desc_lower]
        
        if found_emotions:
            print(f"❌ 包含情感词汇: {found_emotions}")
        else:
            print(f"✅ 不包含直接情感词汇（符合论文要求）")

print("\n" + "="*60)
