#!/usr/bin/env python3
"""
诊断MER-Factory AU数据
"""

import json
import os
from pathlib import Path

def diagnose():
    mer_factory_output = '/home/project/MER-Factory/output'
    
    print("="*60)
    print("MER-Factory AU Data Diagnosis")
    print("="*60)
    
    # 1. 检查目录
    print(f"\n1. Output directory: {mer_factory_output}")
    print(f"   Exists: {os.path.exists(mer_factory_output)}")
    
    # 2. 列出前5个子目录
    subdirs = [d for d in os.listdir(mer_factory_output) if os.path.isdir(os.path.join(mer_factory_output, d))]
    print(f"\n2. Found {len(subdirs)} subdirectories")
    print(f"   First 5: {subdirs[:5]}")
    
    # 3. 检查第一个子目录的内容
    if subdirs:
        first_dir = os.path.join(mer_factory_output, subdirs[0])
        files = os.listdir(first_dir)
        print(f"\n3. Files in {subdirs[0]}:")
        for f in files:
            print(f"   - {f}")
        
        # 4. 查找AU JSON文件
        au_json_files = [f for f in files if '_au_analysis.json' in f]
        print(f"\n4. AU analysis JSON files: {len(au_json_files)}")
        if au_json_files:
            print(f"   Files: {au_json_files}")
        else:
            print(f"   ⚠️ No *_au_analysis.json files found!")
            print(f"   Available files: {files}")
        
        # 5. 如果有JSON文件，检查内容
        if au_json_files:
            json_path = os.path.join(first_dir, au_json_files[0])
            print(f"\n5. Checking JSON structure: {json_path}")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"   Keys: {list(data.keys())}")
                
                if 'fine_grained_descriptions' in data:
                    print(f"   ✅ Has 'fine_grained_descriptions'")
                    print(f"   Frames: {list(data['fine_grained_descriptions'].keys())[:5]}")
                else:
                    print(f"   ❌ Missing 'fine_grained_descriptions'")
                    print(f"   Available keys: {list(data.keys())}")
                
                if 'au_detections' in data:
                    print(f"   ✅ Has 'au_detections'")
                    print(f"   Frames: {list(data['au_detections'].keys())[:5]}")
                else:
                    print(f"   ❌ Missing 'au_detections'")
                
                # 显示一个示例
                if 'fine_grained_descriptions' in data:
                    frame_key = list(data['fine_grained_descriptions'].keys())[0]
                    print(f"\n6. Sample data (frame {frame_key}):")
                    if 'au_detections' in data and frame_key in data['au_detections']:
                        print(f"   AU values: {data['au_detections'][frame_key]}")
                    print(f"   Description: {data['fine_grained_descriptions'][frame_key][:100]}...")
                    
            except Exception as e:
                print(f"   ❌ Error reading JSON: {e}")
    
    # 7. 使用glob查找所有AU JSON
    print(f"\n7. Using glob to find all *_au_analysis.json files:")
    mer_factory_path = Path(mer_factory_output)
    au_json_files = list(mer_factory_path.rglob('*_au_analysis.json'))
    print(f"   Found {len(au_json_files)} files")
    if au_json_files:
        print(f"   First 5:")
        for f in au_json_files[:5]:
            print(f"     - {f}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    diagnose()
