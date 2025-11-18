#!/usr/bin/env python3
"""检查Label CSV文件格式"""

import pandas as pd

csv_path = '/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/track2_train_mercaptionplus_test.csv'

print("="*60)
print("Label CSV文件格式检查")
print("="*60)

# 读取CSV
df = pd.read_csv(csv_path)

print(f"\n✅ 总样本数: {len(df)}")
print(f"\n列名:")
for col in df.columns:
    print(f"  - {col}")

print(f"\n前5行数据:")
print(df.head())

print(f"\n示例数据:")
for i in range(min(3, len(df))):
    print(f"\n样本 {i+1}:")
    for col in df.columns:
        print(f"  {col}: {df.iloc[i][col]}")

# 检查是否有情感标签列
possible_label_cols = ['emotion', 'label', 'sentiment', 'category']
print(f"\n情感标签列:")
for col in possible_label_cols:
    if col in df.columns:
        print(f"  ✅ {col}")
        print(f"     唯一值: {df[col].unique()[:10]}")
    else:
        print(f"  ❌ {col} (不存在)")

print("\n" + "="*60)
