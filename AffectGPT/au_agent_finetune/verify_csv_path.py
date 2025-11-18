#!/usr/bin/env python3
"""验证CSV文件路径是否正确"""

import os

csv_path = '/home/project/MER-Factory/data-process/track2_train_mercaptionplus_test.csv'

print("="*60)
print("验证CSV文件路径")
print("="*60)

print(f"\n检查路径: {csv_path}")
print(f"文件存在: {os.path.exists(csv_path)}")

if os.path.exists(csv_path):
    print(f"✅ 文件找到")
    print(f"文件大小: {os.path.getsize(csv_path)} bytes")
else:
    print(f"❌ 文件不存在")
    
    # 尝试查找可能的位置
    possible_dirs = [
        '/home/project/MER-Factory/data-process',
        '/home/project/MER-Factory/data',
        '/home/project/MER-Factory',
    ]
    
    print(f"\n尝试查找CSV文件:")
    for dir_path in possible_dirs:
        if os.path.exists(dir_path):
            print(f"\n目录 {dir_path} 的内容:")
            files = os.listdir(dir_path)
            csv_files = [f for f in files if f.endswith('.csv')]
            if csv_files:
                for f in csv_files:
                    print(f"  - {f}")
            else:
                print(f"  (没有CSV文件)")

print("\n" + "="*60)
