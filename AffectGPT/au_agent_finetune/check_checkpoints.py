#!/usr/bin/env python3
"""检查训练输出的checkpoint"""

import os
from pathlib import Path

output_dir = '../output/au_agent_qwen2.5_7b_lora'

print("="*60)
print("检查AU Agent训练输出")
print("="*60)

abs_path = os.path.abspath(output_dir)
print(f"\n输出目录: {abs_path}")
print(f"目录存在: {os.path.exists(output_dir)}")

if os.path.exists(output_dir):
    print(f"\n目录内容:")
    for item in sorted(os.listdir(output_dir)):
        item_path = os.path.join(output_dir, item)
        if os.path.isdir(item_path):
            print(f"  📁 {item}/")
            # 列出子目录内容
            try:
                subitems = os.listdir(item_path)
                for subitem in subitems[:5]:  # 只显示前5个
                    print(f"      - {subitem}")
                if len(subitems) > 5:
                    print(f"      ... ({len(subitems)} files total)")
            except:
                pass
        else:
            file_size = os.path.getsize(item_path)
            print(f"  📄 {item} ({file_size} bytes)")
    
    # 检查关键文件
    print(f"\n关键文件检查:")
    key_files = [
        'adapter_model.safetensors',
        'adapter_config.json',
        'trainer_state.json',
    ]
    for f in key_files:
        path = os.path.join(output_dir, f)
        exists = os.path.exists(path)
        print(f"  {'✅' if exists else '❌'} {f}")
else:
    print(f"\n❌ 输出目录不存在")
    
    # 检查父目录
    parent_dir = '../output'
    if os.path.exists(parent_dir):
        print(f"\n父目录 {parent_dir} 存在，内容:")
        for item in os.listdir(parent_dir):
            print(f"  - {item}")

print("\n" + "="*60)
