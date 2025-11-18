#!/usr/bin/env python3
"""
使用CLIP对MER-Factory输出的fine_grained_descriptions进行编码
"""
import os
import json
import clip
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console
from rich.progress import Progress, TaskID, BarColumn, TextColumn, MofNCompleteColumn

console = Console()

def load_clip_model(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    """加载CLIP模型"""
    console.print(f"🔧 Loading CLIP model on device: [yellow]{device}[/yellow]")
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, device

def find_au_analysis_files(output_dir: str) -> List[Tuple[str, str]]:
    """查找所有的au_analysis.json文件"""
    output_path = Path(output_dir)
    if not output_path.exists():
        console.print(f"❌ Output directory not found: {output_dir}")
        return []
    
    files_found = []
    for subfolder in output_path.iterdir():
        if subfolder.is_dir():
            # 查找*_au_analysis.json文件
            for json_file in subfolder.glob("*_au_analysis.json"):
                files_found.append((subfolder.name, str(json_file)))
                
    console.print(f"📁 Found {len(files_found)} AU analysis files")
    return files_found

def extract_fine_grained_descriptions(json_file_path: str) -> Dict[str, str]:
    """从AU分析JSON文件中提取fine_grained_descriptions"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        fine_grained_descriptions = data.get('fine_grained_descriptions', {})
        return fine_grained_descriptions
    
    except Exception as e:
        console.print(f"❌ Error reading {json_file_path}: {e}")
        return {}

def encode_descriptions_with_clip(
    descriptions: Dict[str, str], 
    model, 
    device: str
) -> Dict[str, np.ndarray]:
    """使用CLIP编码描述文本"""
    encoded_features = {}
    
    if not descriptions:
        return encoded_features
    
    # 批量处理所有描述
    frame_indices = list(descriptions.keys())
    texts = list(descriptions.values())
    
    # 使用CLIP的文本编码器
    text_tokens = clip.tokenize(texts).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        # 归一化特征向量
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features = text_features.cpu().numpy()
    
    # 将编码结果映射回frame index
    for i, frame_idx in enumerate(frame_indices):
        encoded_features[frame_idx] = text_features[i]
    
    return encoded_features

def save_encoded_features(
    encoded_features: Dict[str, np.ndarray], 
    video_id: str, 
    output_file: str
):
    """保存编码后的特征"""
    # 转换numpy数组为列表以便JSON序列化
    serializable_features = {}
    for frame_idx, features in encoded_features.items():
        serializable_features[frame_idx] = {
            'features': features.tolist(),
            'shape': features.shape,
            'dtype': str(features.dtype)
        }
    
    save_data = {
        'video_id': video_id,
        'clip_model': 'ViT-B/32',
        'feature_dim': 512,  # ViT-B/32的特征维度
        'encoded_fine_grained_descriptions': serializable_features
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    console.print(f"💾 Saved encoded features to: [green]{output_file}[/green]")

def process_all_files(mer_factory_output_dir: str, affectgpt_output_dir: str):
    """处理所有AU分析文件"""
    # 创建输出目录
    output_path = Path(affectgpt_output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 加载CLIP模型
    model, device = load_clip_model()
    
    # 查找所有AU分析文件
    au_files = find_au_analysis_files(mer_factory_output_dir)
    
    if not au_files:
        console.print("❌ No AU analysis files found!")
        return
    
    # 处理每个文件
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        refresh_per_second=2,
    ) as progress:
        
        task = progress.add_task("🔄 Processing AU analysis files", total=len(au_files))
        
        for video_id, json_file_path in au_files:
            progress.update(task, description=f"Processing {video_id}")
            
            # 提取fine_grained_descriptions
            descriptions = extract_fine_grained_descriptions(json_file_path)
            
            if descriptions:
                # 使用CLIP编码
                encoded_features = encode_descriptions_with_clip(descriptions, model, device)
                
                # 保存编码结果
                output_file = output_path / f"{video_id}_clip_features.json"
                save_encoded_features(encoded_features, video_id, str(output_file))
                
                console.print(f"✅ Processed {video_id}: {len(descriptions)} descriptions encoded")
            else:
                console.print(f"⚠️  No fine_grained_descriptions found in {video_id}")
            
            progress.advance(task)
        
        progress.update(task, description="✅ All files processed")

def main():
    """主函数"""
    console.rule("[bold blue]🎯 CLIP Encoding for Fine-Grained Descriptions[/bold blue]")
    
    # 配置路径
    mer_factory_output = "G:/Project/MER-Factory/output"
    affectgpt_output = "G:/Project/AffectGPT/AffectGPT/clip_encoded_features"
    
    console.print(f"📂 Input directory: [cyan]{mer_factory_output}[/cyan]")
    console.print(f"📂 Output directory: [cyan]{affectgpt_output}[/cyan]")
    
    # 检查CLIP是否可用
    try:
        import clip
        console.print("✅ CLIP module imported successfully")
    except ImportError:
        console.print("❌ CLIP not installed. Please run: pip install git+https://github.com/openai/CLIP.git")
        return
    
    # 处理所有文件
    process_all_files(mer_factory_output, affectgpt_output)
    
    console.rule("[bold green]✨ Encoding Complete![/bold green]")

if __name__ == "__main__":
    main()
