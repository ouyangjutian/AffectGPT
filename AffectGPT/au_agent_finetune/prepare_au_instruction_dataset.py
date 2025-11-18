#!/usr/bin/env python3
"""
准备AU Agent微调数据集
从MER-Factory的AU分析结果构建指令微调数据
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import random
import pandas as pd
import ast


def load_label_mapping(csv_path: str) -> Dict[str, str]:
    """加载情感标签映射
    
    Args:
        csv_path: CSV文件路径
    
    Returns:
        {video_name: emotion_labels} 字典
    """
    df = pd.read_csv(csv_path)
    label_map = {}
    
    for _, row in df.iterrows():
        video_name = row['name']
        openset = row['openset']
        
        # 解析openset（可能是字符串格式的列表）
        if isinstance(openset, str):
            try:
                # 尝试解析为列表
                labels = ast.literal_eval(openset)
            except:
                # 如果解析失败，使用原始字符串
                labels = [openset]
        else:
            labels = [openset]
        
        # 使用第一个标签作为主要情感，或者拼接所有标签
        # 方案1: 只用第一个标签
        # emotion = labels[0] if labels else 'neutral'
        
        # 方案2: 拼接所有标签（更丰富）
        emotion = ', '.join(labels) if labels else 'neutral'
        
        label_map[video_name] = emotion
    
    return label_map


def load_au_analysis(json_path: str) -> Dict:
    """加载AU分析JSON"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def create_instruction_sample(au_result: Dict, frame_idx: str, label_map: Dict[str, str] = None, video_name: str = None) -> Dict:
    """
    从AU分析结果创建指令微调样本
    
    格式：
    {
        "instruction": "Based on the following Action Unit detections, describe the facial expression:",
        "input": "AU01: 0.98, AU05: 0.98, AU07: 2.35, AU25: 1.76",
        "output": "The facial expression exhibits subtle brow lowering, neutral ocular engagement with mild lid tightening, and slight lip parting, consistent with a prototypical neutral state."
    }
    """
    # 获取fine-grained描述
    description = au_result.get('fine_grained_descriptions', {}).get(frame_idx, "")
    if not description:
        return None
    
    # 从per_frame_au_descriptions找到对应帧的AU值和AU描述
    per_frame_data = au_result.get('per_frame_au_descriptions', [])
    au_values = None
    au_description = None
    
    # 查找匹配的帧
    frame_num = int(frame_idx)
    for frame_data in per_frame_data:
        if frame_data.get('frame') == frame_num:
            au_values = frame_data.get('active_aus', {})
            au_description = frame_data.get('au_description', None)  # 提取AU描述
            break
    
    if not au_values:
        return None
    
    # 构建AU数值文本（移除_r后缀）
    au_values_text = ", ".join([f"{au_id.replace('_r', '')}: {value:.2f}" for au_id, value in au_values.items()])
    
    # 获取情感标签
    emotion_label = None
    if label_map and video_name:
        emotion_label = label_map.get(video_name, None)
    
    # Prompt模板（论文中的Tp）
    # 推荐选项: 强调情感引导和完整输入（更符合论文方法）
    prompt_tp = "Given the emotion label, AU intensity values, and their semantic descriptions, provide a detailed and natural facial expression description:"
    
    # 备选Prompt模板（可以取消注释使用随机选择增强泛化）:
    # prompt_templates = [
    #     "Given the emotion label, AU intensity values, and their semantic descriptions, provide a detailed and natural facial expression description:",
    #     "Describe the facial expression by analyzing the emotion context and Action Unit activations:",
    #     "Based on the provided emotion label and Action Unit data (including intensity values and semantic meanings), generate a comprehensive facial expression description:",
    #     "Describe the facial expression using the emotion label and AU detections provided:",
    # ]
    # prompt_tp = random.choice(prompt_templates)
    
    # 根据是否有情感标签构建输入格式
    if emotion_label:
        # 有情感标签：完整的论文方法 = Label + Prompt (Tp) + AU values + AU descriptions
        # LLaMA-Factory的instruction字段用于任务级别的指令
        instruction = "Generate a detailed facial expression description based on the given information."
        
        # input包含：Label + Prompt + AU values + AU descriptions（完全符合论文图3a）
        if au_description:
            # 有AU描述：完整输入
            input_text = f"""Emotion: {emotion_label}
Prompt: {prompt_tp}
AU values: {au_values_text}
AU descriptions: {au_description}"""
        else:
            # 无AU描述：回退到只有AU值
            input_text = f"""Emotion: {emotion_label}
Prompt: {prompt_tp}
AU detections: {au_values_text}"""
    else:
        # 无情感标签：回退到 Prompt + AUs
        instruction = "Generate a facial expression description based on AU detections."
        if au_description:
            input_text = f"""Prompt: {prompt_tp}
AU values: {au_values_text}
AU descriptions: {au_description}"""
        else:
            input_text = f"""Prompt: {prompt_tp}
AU detections: {au_values_text}"""
    
    sample = {
        "instruction": instruction,
        "input": input_text,
        "output": description
    }
    
    return sample


def process_mer_factory_outputs(
    mer_factory_output_dir: str,
    output_json_path: str,
    label_csv_path: str = None,
    max_samples: int = None
):
    """
    处理MER-Factory输出，构建AU指令数据集
    
    Args:
        mer_factory_output_dir: MER-Factory输出目录
        output_json_path: 输出的指令数据集JSON路径
        label_csv_path: 情感标签CSV文件路径（可选，如果提供则使用Label+AU模式）
        max_samples: 最大样本数（None=全部）
    """
    print("="*60)
    print("AU Instruction Dataset Preparation")
    print("="*60)
    
    # 加载情感标签映射（如果提供）
    label_map = None
    if label_csv_path and os.path.exists(label_csv_path):
        print(f"\n📋 Loading emotion labels from: {label_csv_path}")
        label_map = load_label_mapping(label_csv_path)
        print(f"✅ Loaded {len(label_map)} emotion labels")
    else:
        print(f"\n⚠️ No emotion label file provided, using AU-only mode")
    
    # 查找所有AU分析JSON
    mer_factory_path = Path(mer_factory_output_dir)
    au_json_files = list(mer_factory_path.rglob('*_au_analysis.json'))
    
    print(f"\n📁 Found {len(au_json_files)} AU analysis files")
    
    # 收集所有指令样本
    all_samples = []
    
    for json_file in au_json_files:
        try:
            au_result = load_au_analysis(json_file)
            
            # 提取video名称（从文件路径或JSON中的source_path）
            video_name = json_file.stem.replace('_au_analysis', '')
            
            # 为每一帧创建样本
            for frame_idx in au_result.get('fine_grained_descriptions', {}).keys():
                sample = create_instruction_sample(au_result, frame_idx, label_map, video_name)
                if sample:
                    all_samples.append(sample)
            
            if len(all_samples) % 1000 == 0:
                print(f"  Processed {len(all_samples)} samples...")
                
        except Exception as e:
            print(f"⚠️ Error processing {json_file}: {e}")
            continue
    
    print(f"\n✅ Total samples collected: {len(all_samples)}")
    
    # 限制样本数量
    if max_samples and len(all_samples) > max_samples:
        print(f"📊 Sampling {max_samples} from {len(all_samples)} samples")
        all_samples = random.sample(all_samples, max_samples)
    
    # 划分训练集和验证集
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.95)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]
    
    print(f"📊 Train: {len(train_samples)}, Val: {len(val_samples)}")
    
    # 保存数据集
    dataset = {
        "train": train_samples,
        "validation": val_samples,
        "metadata": {
            "total_samples": len(all_samples),
            "source": "MER-Factory AU Analysis",
            "format": "instruction_following",
            "description": "AU detection results to natural language descriptions"
        }
    }
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Dataset saved to: {output_json_path}")
    
    # 显示样本示例
    print("\n" + "="*60)
    print("Sample Examples:")
    print("="*60)
    for i, sample in enumerate(train_samples[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"Instruction: {sample['instruction']}")
        print(f"Input: {sample['input']}")
        print(f"Output: {sample['output'][:150]}...")


def convert_to_llama_factory_format(
    instruction_dataset_path: str,
    output_train_jsonl: str,
    output_val_jsonl: str = None
):
    """
    转换为LLaMA-Factory格式
    
    格式：每行一个JSON，包含instruction、input、output
    """
    with open(instruction_dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    train_samples = dataset['train']
    val_samples = dataset.get('validation', [])
    
    # 保存训练集JSONL
    with open(output_train_jsonl, 'w', encoding='utf-8') as f:
        for sample in train_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    print(f"✅ Train JSONL saved to: {output_train_jsonl}")
    
    # 保存验证集JSONL（如果有）
    if output_val_jsonl and val_samples:
        with open(output_val_jsonl, 'w', encoding='utf-8') as f:
            for sample in val_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        print(f"✅ Val JSONL saved to: {output_val_jsonl}")
    
    return len(train_samples), len(val_samples)


if __name__ == '__main__':
    # 配置
    MER_FACTORY_OUTPUT = '/home/project/MER-Factory/output'
    LABEL_CSV_PATH = '/home/project/Dataset/Emotion/MER2025/dataset/mer2025-dataset/track2_train_mercaptionplus_test.csv'  # 情感标签CSV
    OUTPUT_JSON = './au_instruction_dataset.json'  # 保存在au_agent_finetune文件夹
    OUTPUT_TRAIN_JSONL = './au_instruction_dataset_train.jsonl'  # 训练集
    OUTPUT_VAL_JSONL = './au_instruction_dataset_val.jsonl'      # 验证集
    
    # 准备数据集
    process_mer_factory_outputs(
        mer_factory_output_dir=MER_FACTORY_OUTPUT,
        label_csv_path=LABEL_CSV_PATH,
        output_json_path=OUTPUT_JSON,
        max_samples=100000  # 限制10万样本
    )
    
    # 转换为LLaMA-Factory格式（生成train和val两个文件）
    train_count, val_count = convert_to_llama_factory_format(
        OUTPUT_JSON, 
        OUTPUT_TRAIN_JSONL,
        OUTPUT_VAL_JSONL
    )
    
    print("\n" + "="*60)
    print("Dataset preparation complete!")
    print("="*60)
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Train samples: {train_count}")
    print(f"  - Val samples: {val_count}")
    print(f"\n📁 Generated Files:")
    print(f"  - Full dataset: {OUTPUT_JSON}")
    print(f"  - Train JSONL: {OUTPUT_TRAIN_JSONL}")
    print(f"  - Val JSONL: {OUTPUT_VAL_JSONL}")
    print(f"\n🚀 Next steps:")
    print(f"1. Review dataset quality")
    print(f"2. Run training: bash train_au_agent.sh")
    print(f"3. Test AU Agent generation quality")
