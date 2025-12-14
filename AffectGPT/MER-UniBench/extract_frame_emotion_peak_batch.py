#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为MER-UniBench 9个数据集批量预提取emotion_peak采样的Frame特征
避免推理时的实时文件I/O开销

支持数据集:
- CMUMOSEI, CMUMOSI, IEMOCAP, MELD
- MER2023, MER2024
- OVMERDPLUS, SIMS, SIMSv2
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

# 添加路径（脚本在MER-UniBench子目录，需要添加上级目录）
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)  # AffectGPT根目录
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'my_affectgpt'))

from my_affectgpt.common.registry import registry
from my_affectgpt.processors.video_processor import load_video
import config


# 数据集配置（基于MER-Factory的batch_extract_au_multi_datasets.py配置）
# Linux服务器路径
DATASET_ROOT = "/home/project/Dataset/Emotion/MER2025/dataset"

DATASET_CONFIGS = {
    'cmumosei': {
        'video_root': f'{DATASET_ROOT}/cmumosei-process/subvideo_new',
        'label_file': f'{DATASET_ROOT}/cmumosei-process/label.npz',
        'label_type': 'npz',
        'corpus_key': 'test_corpus',  # 推理使用测试集
        'video_ext': '.mp4',
        'mer_factory': 'CMUMOSEI'
    },
    'cmumosi': {
        'video_root': f'{DATASET_ROOT}/cmumosi-process/subvideo',
        'label_file': f'{DATASET_ROOT}/cmumosi-process/label.npz',
        'label_type': 'npz',
        'corpus_key': 'test_corpus',
        'video_ext': '.mp4',
        'mer_factory': 'CMUMOSI'
    },
    'iemocap': {
        'video_root': f'{DATASET_ROOT}/iemocap-process/subvideo-tgt',
        'label_file': f'{DATASET_ROOT}/iemocap-process/label_4way.npz',
        'label_type': 'npz',
        'corpus_key': 'whole_corpus',  # IEMOCAP特殊：使用whole_corpus
        'session_filter': 'Ses05',  # 只处理Ses05（测试集）
        'video_ext': '.avi',
        'mer_factory': 'IEMOCAPFour'
    },
    'meld': {
        'video_root': f'{DATASET_ROOT}/meld-process/subvideo',
        'label_file': f'{DATASET_ROOT}/meld-process/label.npz',
        'label_type': 'npz',
        'corpus_key': 'test_corpus',
        'video_ext': '.mp4',
        'mer_factory': 'MELD'
    },
    'mer2023': {
        'video_root': f'{DATASET_ROOT}/mer2023-dataset-process/video',
        'label_file': f'{DATASET_ROOT}/mer2023-dataset-process/label-6way.npz',
        'label_type': 'npz',
        'corpus_key': 'test1_corpus',  # 推理使用test1
        'video_ext': '.mp4',
        'mer_factory': 'MER2023'
    },
    'mer2024': {
        'video_root': f'{DATASET_ROOT}/mer2024-dataset-process/video',
        'label_file': f'{DATASET_ROOT}/mer2024-dataset-process/label-6way.npz',
        'label_type': 'npz',
        'corpus_key': 'test1_corpus',
        'video_ext': '.mp4',
        'mer_factory': 'MER2024'
    },
    'ovmerdplus': {
        'video_root': f'{DATASET_ROOT}/ovmerdplus-process/video',
        'label_file': f'{DATASET_ROOT}/ovmerdplus-process/subtitle_eng.csv',  # 测试集用subtitle_eng.csv
        'label_type': 'csv',
        'name_column': 'name',
        'video_ext': '.mp4',
        'mer_factory': 'OVMERDPlus'
    },
    'sims': {
        'video_root': f'{DATASET_ROOT}/sims-process/video',
        'label_file': f'{DATASET_ROOT}/sims-process/label.npz',
        'label_type': 'npz',
        'corpus_key': 'test_corpus',
        'video_ext': '.mp4',
        'mer_factory': 'SIMS'
    },
    'simsv2': {
        'video_root': f'{DATASET_ROOT}/simsv2-process/video_new',
        'label_file': f'{DATASET_ROOT}/simsv2-process/label.npz',
        'label_type': 'npz',
        'corpus_key': 'test_corpus',
        'video_ext': '.mp4',
        'mer_factory': 'SIMSv2'
    }
}


def load_sample_names(dataset_name, label_file, dataset_config):
    """从标签文件加载样本名称列表
    
    Args:
        dataset_name: 数据集名称
        label_file: 标签文件路径
        dataset_config: 数据集配置（包含label_type, corpus_key等）
    """
    label_type = dataset_config.get('label_type', 'json')
    
    if label_type == 'csv' or label_file.endswith('.csv'):
        # CSV格式（如OVMERDPlus）
        import pandas as pd
        df = pd.read_csv(label_file)
        name_column = dataset_config.get('name_column', 'name')
        samples = df[name_column].dropna().tolist()
    elif label_type == 'npz' or label_file.endswith('.npz'):
        # NPZ格式
        data = np.load(label_file, allow_pickle=True)
        corpus_key = dataset_config.get('corpus_key', 'test_corpus')
        
        if corpus_key in data:
            corpus_data = data[corpus_key].item()
            if isinstance(corpus_data, dict):
                all_samples = list(corpus_data.keys())
                
                # IEMOCAP特殊处理：通过session过滤
                if dataset_name == 'iemocap' and 'session_filter' in dataset_config:
                    session_filter = dataset_config['session_filter']
                    samples = [s for s in all_samples if s.startswith(session_filter)]
                else:
                    samples = all_samples
            else:
                samples = []
        else:
            print(f"⚠️  Warning: corpus_key '{corpus_key}' not found in {label_file}")
            samples = []
    elif label_file.endswith('.json'):
        # JSON格式
        with open(label_file, 'r') as f:
            data = json.load(f)
        if 'test' in data:
            samples = list(data['test'].keys())
        else:
            samples = list(data.keys())
    else:
        raise ValueError(f"Unsupported label file format: {label_file}")
    
    return samples


def extract_frame_features_emotion_peak(
    dataset_name,
    video_root,
    sample_names,
    output_dir,
    visual_encoder,
    mer_factory_output,
    n_frms=8,
    device='cuda:0',
    quiet=False
):
    """
    提取emotion_peak采样的Frame特征
    
    Args:
        dataset_name: 数据集名称
        video_root: 视频根目录
        sample_names: 样本名称列表
        output_dir: 输出目录
        visual_encoder: 视觉编码器实例
        mer_factory_output: MER-Factory输出根目录
        n_frms: 采样帧数
        device: 设备
        quiet: 是否静默模式
    """
    os.makedirs(output_dir, exist_ok=True)
    
    success_count = 0
    skip_count = 0
    error_count = 0
    errors = []
    
    # 构建MER-Factory数据集路径
    mer_factory_dataset_path = os.path.join(mer_factory_output, DATASET_CONFIGS[dataset_name]['mer_factory'])
    
    for sample_name in tqdm(sample_names, desc=f"Extracting {dataset_name} Frame (emotion_peak)", disable=quiet):
        output_file = os.path.join(output_dir, f'{sample_name}.npy')
        
        # 跳过已存在的文件
        if os.path.exists(output_file):
            skip_count += 1
            continue
        
        try:
            # 查找视频文件
            video_path = None
            for ext in ['.mp4', '.avi', '.mkv', '.mov']:
                candidate = os.path.join(video_root, f'{sample_name}{ext}')
                if os.path.exists(candidate):
                    video_path = candidate
                    break
            
            if not video_path:
                raise FileNotFoundError(f"Video not found for {sample_name}")
            
            # 使用emotion_peak采样加载视频帧
            # 传递video_name和mer_factory_output以启用智能采样
            raw_frames, msg = load_video(
                video_path=video_path,
                n_frms=n_frms,
                height=224,
                width=224,
                sampling='emotion_peak',  # 🎯 关键：使用emotion_peak采样
                return_msg=True,
                video_name=sample_name,  # 传递样本名
                mer_factory_output=mer_factory_dataset_path  # 传递MER-Factory路径
            )
            
            # 转换为CLIP格式 [C, T, H, W]
            if raw_frames.dim() == 4:  # [C, T, H, W]
                frames = raw_frames.unsqueeze(0).to(device)  # [1, C, T, H, W]
            else:
                raise ValueError(f"Unexpected frame shape: {raw_frames.shape}")
            
            # 提取特征
            with torch.no_grad():
                # CLIP编码器输出: [1, T, 768]
                features = visual_encoder(frames, frames)  # (video, raw_video)
                
                # 转为numpy并去掉batch维度
                features_np = features.squeeze(0).cpu().numpy()  # [T, 768]
            
            # 保存特征
            np.save(output_file, features_np)
            success_count += 1
            
        except Exception as e:
            error_count += 1
            error_msg = f"{sample_name}: {str(e)}"
            errors.append(error_msg)
            if not quiet and error_count <= 5:  # 只打印前5个错误
                print(f"  ⚠️  {error_msg}")
    
    # 打印统计信息
    if not quiet:
        print(f"\n{'='*70}")
        print(f"✅ {dataset_name.upper()} Frame (emotion_peak) Extraction Complete")
        print(f"{'='*70}")
        print(f"  Success: {success_count}")
        print(f"  Skipped: {skip_count} (already exists)")
        print(f"  Errors:  {error_count}")
        if error_count > 0:
            print(f"\n  First {min(5, len(errors))} errors:")
            for err in errors[:5]:
                print(f"    - {err}")
        print(f"{'='*70}\n")
    
    return success_count, skip_count, error_count


def main():
    parser = argparse.ArgumentParser(description='批量预提取MER-UniBench数据集的emotion_peak Frame特征')
    parser.add_argument('--datasets', type=str, nargs='+', 
                       default=['cmumosei', 'cmumosi', 'iemocap', 'meld', 'mer2023', 'mer2024', 'ovmerdplus', 'sims', 'simsv2'],
                       help='要处理的数据集列表')
    parser.add_argument('--output-root', type=str, default='./preextracted_features',
                       help='特征输出根目录')
    parser.add_argument('--mer-factory-output', type=str, default='/home/project/MER-Factory/output',
                       help='MER-Factory输出根目录（包含au_info）')
    parser.add_argument('--visual-encoder', type=str, default='CLIP_VIT_LARGE',
                       help='视觉编码器名称')
    parser.add_argument('--n-frms', type=int, default=8,
                       help='采样帧数')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='计算设备')
    parser.add_argument('--quiet', action='store_true',
                       help='静默模式')
    
    args = parser.parse_args()
    
    # 检查MER-Factory路径
    if not os.path.exists(args.mer_factory_output):
        print(f"❌ MER-Factory output directory not found: {args.mer_factory_output}")
        print(f"   emotion_peak采样需要MER-Factory生成的au_info")
        print(f"   请先运行MER-Factory处理这些数据集")
        return
    
    print(f"\n{'='*70}")
    print("🚀 MER-UniBench Frame Emotion_Peak Feature Extraction")
    print(f"{'='*70}\n")
    print(f"📊 Datasets: {', '.join(args.datasets)}")
    print(f"📁 Output root: {args.output_root}")
    print(f"🎯 Sampling strategy: emotion_peak (based on au_info)")
    print(f"📂 MER-Factory output: {args.mer_factory_output}")
    print(f"🔧 Visual encoder: {args.visual_encoder}")
    print(f"🎬 Frames per sample: {args.n_frms}")
    print(f"💻 Device: {args.device}\n")
    
    # 加载视觉编码器（只需加载一次）
    print("🔧 Loading Visual Encoder...")
    encoder_cls = registry.get_visual_encoder_class(args.visual_encoder)
    visual_encoder = encoder_cls().to(args.device)
    visual_encoder.eval()
    print("✅ Visual Encoder loaded\n")
    
    # 处理每个数据集
    total_stats = {'success': 0, 'skip': 0, 'error': 0}
    
    for dataset_name in args.datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"⚠️  Unknown dataset: {dataset_name}, skipping...")
            continue
        
        config = DATASET_CONFIGS[dataset_name]
        
        # 检查数据集路径
        if not os.path.exists(config['video_root']):
            print(f"⚠️  Video root not found for {dataset_name}: {config['video_root']}, skipping...")
            continue
        
        if not os.path.exists(config['label_file']):
            print(f"⚠️  Label file not found for {dataset_name}: {config['label_file']}, skipping...")
            continue
        
        # 加载样本名称
        sample_names = load_sample_names(dataset_name, config['label_file'], config)
        
        # 构建输出目录
        output_dir = os.path.join(
            args.output_root,
            dataset_name,
            f'frame_{args.visual_encoder}_emotion_peak_{args.n_frms}frms'
        )
        
        print(f"\n{'='*70}")
        print(f"📦 Processing {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"  Video root: {config['video_root']}")
        print(f"  Samples: {len(sample_names)}")
        print(f"  Output: {output_dir}")
        print(f"  MER-Factory: {os.path.join(args.mer_factory_output, config['mer_factory'])}\n")
        
        # 提取特征
        success, skip, error = extract_frame_features_emotion_peak(
            dataset_name=dataset_name,
            video_root=config['video_root'],
            sample_names=sample_names,
            output_dir=output_dir,
            visual_encoder=visual_encoder,
            mer_factory_output=args.mer_factory_output,
            n_frms=args.n_frms,
            device=args.device,
            quiet=args.quiet
        )
        
        total_stats['success'] += success
        total_stats['skip'] += skip
        total_stats['error'] += error
    
    # 打印总体统计
    print(f"\n{'='*70}")
    print("🎉 All Datasets Processed")
    print(f"{'='*70}")
    print(f"  Total Success: {total_stats['success']}")
    print(f"  Total Skipped: {total_stats['skip']}")
    print(f"  Total Errors:  {total_stats['error']}")
    print(f"{'='*70}\n")
    
    print("💡 Usage:")
    print("   在推理配置中设置:")
    print("   - frame_sampling: 'emotion_peak'")
    print("   - use_preextracted_features: True")
    print("   - preextracted_root: './preextracted_features/<dataset_name>'")
    print()


if __name__ == '__main__':
    main()
