#!/usr/bin/env python3
"""
批量生成AU描述
用于训练前预生成所有样本的AU描述，避免训练时重复调用LLM
"""

import sys
sys.path.append('..')

from my_affectgpt.models.au_agent import AUAgent
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
import ast

# 配置
BASE_MODEL = "/home/project/Dataset/Emotion/tools/transformer/LLM/Qwen2.5-7B-Instruct"
LORA_WEIGHTS = "/home/project/AffectGPT/AffectGPT/output/au_agent_qwen2.5_7b_lora"
MER_FACTORY_OUTPUT = "/home/project/MER-Factory/output"
LABEL_CSV = "/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/track2_train_mercaptionplus_test.csv"
OUTPUT_JSON = "./au_descriptions_all.json"

def load_emotion_labels(csv_path):
    """加载情感标签"""
    df = pd.read_csv(csv_path)
    label_map = {}
    
    for _, row in df.iterrows():
        video_name = row['name']
        if 'openset' in row:
            try:
                openset = ast.literal_eval(row['openset'])
                if isinstance(openset, list) and len(openset) > 0:
                    label_map[video_name] = ', '.join(openset)
            except:
                pass
    
    return label_map

def main():
    print("="*60)
    print("批量生成AU描述")
    print("="*60)
    
    # 1. 加载AU Agent
    print("\n[步骤1] 加载AU Agent...")
    au_agent = AUAgent(
        base_model_path=BASE_MODEL,
        lora_weights_path=LORA_WEIGHTS,
        use_lora=True
    )
    
    # 2. 加载情感标签
    print("\n[步骤2] 加载情感标签...")
    label_map = load_emotion_labels(LABEL_CSV)
    print(f"✅ 加载 {len(label_map)} 个情感标签")
    
    # 3. 找到所有MER-Factory JSON文件
    print("\n[步骤3] 扫描MER-Factory输出...")
    mer_factory_path = Path(MER_FACTORY_OUTPUT)
    au_json_files = list(mer_factory_path.rglob('*_au_analysis.json'))
    print(f"✅ 找到 {len(au_json_files)} 个AU分析文件")
    
    # 4. 批量生成描述
    print("\n[步骤4] 批量生成AU描述...")
    all_descriptions = {}
    
    for json_file in tqdm(au_json_files, desc="生成描述"):
        try:
            # 提取视频名称
            video_name = json_file.stem.replace('_au_analysis', '')
            
            # 加载AU数据
            with open(json_file, 'r') as f:
                au_result = json.load(f)
            
            # 获取情感标签
            emotion_label = label_map.get(video_name, None)
            
            # 为每一帧生成描述
            per_frame_data = au_result.get('per_frame_au_descriptions', [])
            frame_descriptions = {}
            
            for frame_data in per_frame_data:
                frame_idx = frame_data.get('frame')
                au_values = frame_data.get('active_aus', {})
                au_description = frame_data.get('au_description', '')
                
                if not au_values:
                    continue
                
                # 生成描述
                try:
                    description = au_agent.generate_description(
                        au_values=au_values,
                        au_description=au_description,
                        emotion_label=emotion_label,
                        temperature=0.7
                    )
                    frame_descriptions[str(frame_idx)] = description
                except Exception as e:
                    print(f"⚠️ 生成失败 {video_name} frame {frame_idx}: {e}")
                    continue
            
            # 存储该视频的所有帧描述
            all_descriptions[video_name] = {
                'emotion_label': emotion_label,
                'frame_descriptions': frame_descriptions,
                'num_frames': len(frame_descriptions)
            }
            
        except Exception as e:
            print(f"⚠️ 处理失败 {json_file}: {e}")
            continue
    
    # 5. 保存结果
    print(f"\n[步骤5] 保存结果到 {OUTPUT_JSON}...")
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(all_descriptions, f, ensure_ascii=False, indent=2)
    
    # 统计
    total_frames = sum(data['num_frames'] for data in all_descriptions.values())
    print(f"\n✅ 生成完成!")
    print(f"  - 总视频数: {len(all_descriptions)}")
    print(f"  - 总帧数: {total_frames}")
    print(f"  - 平均每视频帧数: {total_frames / len(all_descriptions):.1f}")
    
    # 显示示例
    print(f"\n{'='*60}")
    print("示例描述:")
    print(f"{'='*60}")
    
    for i, (video_name, data) in enumerate(list(all_descriptions.items())[:3], 1):
        print(f"\n--- 示例 {i}: {video_name} ---")
        print(f"Emotion: {data['emotion_label']}")
        
        # 显示第一帧的描述
        first_frame_idx = list(data['frame_descriptions'].keys())[0]
        first_desc = data['frame_descriptions'][first_frame_idx]
        print(f"Frame {first_frame_idx} 描述:")
        print(f"  {first_desc[:150]}...")
    
    print(f"\n{'='*60}")
    print("批量生成完成！")
    print(f"{'='*60}")
    print(f"\n下一步:")
    print(f"1. 检查生成质量: python check_au_descriptions_quality.py")
    print(f"2. 集成到AffectGPT训练: 参考 AU_AGENT_INTEGRATION.md")

if __name__ == '__main__':
    main()
