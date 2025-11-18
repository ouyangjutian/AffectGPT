#!/usr/bin/env python3
"""检查MER-Factory JSON中是否有AU语义描述"""

import json

json_path = '/home/project/MER-Factory/output/samplenew3_00043253/samplenew3_00043253_au_analysis.json'

with open(json_path, 'r') as f:
    data = json.load(f)

print("="*60)
print("检查AU Detection Result的两部分数据")
print("="*60)

if 'per_frame_au_descriptions' in data and len(data['per_frame_au_descriptions']) > 0:
    frame = data['per_frame_au_descriptions'][0]
    
    print(f"\n✅ 第一帧数据 (frame {frame['frame']}):")
    print(f"\n1️⃣ AU数值 (active_aus):")
    for au_id, value in frame.get('active_aus', {}).items():
        print(f"   {au_id}: {value}")
    
    print(f"\n2️⃣ AU语义描述 (au_description):")
    if 'au_description' in frame:
        print(f"   ✅ {frame['au_description']}")
    else:
        print(f"   ❌ 不存在")
        print(f"   可用字段: {list(frame.keys())}")
    
    print(f"\n完整帧数据结构:")
    import json
    print(json.dumps(frame, indent=2, ensure_ascii=False))

print("\n" + "="*60)
