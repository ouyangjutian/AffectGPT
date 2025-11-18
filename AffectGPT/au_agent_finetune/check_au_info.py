#!/usr/bin/env python3
"""检查au_info的实际格式"""

import json

json_path = '/home/project/MER-Factory/output/samplenew3_00043253/samplenew3_00043253_au_analysis.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print("="*60)
print("AU Info Structure Check")
print("="*60)

# 查看au_info
if 'au_info' in data:
    print("\n✅ Found 'au_info'")
    frames = list(data['au_info'].keys())
    print(f"Frames: {frames[:5]}")
    
    # 查看第一帧的结构
    first_frame = frames[0]
    au_info_frame = data['au_info'][first_frame]
    
    print(f"\nFrame '{first_frame}' au_info:")
    print(f"Type: {type(au_info_frame)}")
    
    if isinstance(au_info_frame, dict):
        print(f"Keys: {list(au_info_frame.keys())[:10]}")
        # 显示前几个AU值
        for i, (au_id, value) in enumerate(list(au_info_frame.items())[:5]):
            print(f"  {au_id}: {value}")
    elif isinstance(au_info_frame, list):
        print(f"Length: {len(au_info_frame)}")
        print(f"First item: {au_info_frame[0]}")
    else:
        print(f"Content: {au_info_frame}")

# 查看fine_grained_descriptions
if 'fine_grained_descriptions' in data:
    print("\n✅ Found 'fine_grained_descriptions'")
    frames = list(data['fine_grained_descriptions'].keys())
    print(f"Frames: {frames[:5]}")
    
    first_frame = frames[0]
    description = data['fine_grained_descriptions'][first_frame]
    print(f"\nFrame '{first_frame}' description:")
    print(f"  {description[:200]}...")

print("\n" + "="*60)
