#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
验证AU特征提取流程
检查MER-Factory输出 -> CLIP特征提取 -> 训练加载
"""

import os
import json
import numpy as np
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def check_mer_factory_output(mer_factory_root, sample_names):
    """检查MER-Factory输出的JSON文件"""
    console.print("\n[bold cyan]📂 步骤1: 检查MER-Factory输出[/bold cyan]")
    
    results = []
    for sample_name in sample_names:
        json_path = Path(mer_factory_root) / sample_name / f"{sample_name}_au_analysis.json"
        
        status = {
            'sample': sample_name,
            'exists': json_path.exists(),
            'has_summary': False,
            'num_descriptions': 0,
            'frame_keys': []
        }
        
        if json_path.exists():
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'summary_description' in data:
                    status['has_summary'] = True
                    status['num_descriptions'] = len(data['summary_description'])
                    status['frame_keys'] = list(data['summary_description'].keys())[:5]
                
            except Exception as e:
                console.print(f"[red]❌ 读取失败: {sample_name} - {e}[/red]")
        
        results.append(status)
    
    # 显示结果表格
    table = Table(title="MER-Factory输出检查")
    table.add_column("样本", style="cyan")
    table.add_column("文件存在", style="green")
    table.add_column("summary_description", style="yellow")
    table.add_column("描述数量", style="magenta")
    table.add_column("帧键示例", style="blue")
    
    for r in results:
        table.add_row(
            r['sample'],
            "✅" if r['exists'] else "❌",
            "✅" if r['has_summary'] else "❌",
            str(r['num_descriptions']),
            str(r['frame_keys']) if r['frame_keys'] else "N/A"
        )
    
    console.print(table)
    
    # 统计
    total = len(results)
    valid = sum(1 for r in results if r['exists'] and r['has_summary'])
    console.print(f"\n[green]✅ 有效样本: {valid}/{total}[/green]")
    
    return results


def check_extracted_features(preextracted_root, sample_names):
    """检查提取的CLIP特征"""
    console.print("\n[bold cyan]🔧 步骤2: 检查提取的CLIP特征[/bold cyan]")
    
    feat_dir = Path(preextracted_root) / 'au_CLIP_VITB32_8frms'
    
    if not feat_dir.exists():
        console.print(f"[red]❌ 特征目录不存在: {feat_dir}[/red]")
        return []
    
    results = []
    for sample_name in sample_names:
        feat_path = feat_dir / f"{sample_name}.npy"
        
        status = {
            'sample': sample_name,
            'exists': feat_path.exists(),
            'shape': None,
            'dtype': None,
            'range': None
        }
        
        if feat_path.exists():
            try:
                feat = np.load(feat_path)
                status['shape'] = feat.shape
                status['dtype'] = str(feat.dtype)
                status['range'] = f"[{feat.min():.3f}, {feat.max():.3f}]"
            except Exception as e:
                console.print(f"[red]❌ 加载失败: {sample_name} - {e}[/red]")
        
        results.append(status)
    
    # 显示结果表格
    table = Table(title="CLIP特征检查")
    table.add_column("样本", style="cyan")
    table.add_column("文件存在", style="green")
    table.add_column("形状", style="yellow")
    table.add_column("数据类型", style="magenta")
    table.add_column("值范围", style="blue")
    
    for r in results:
        table.add_row(
            r['sample'],
            "✅" if r['exists'] else "❌",
            str(r['shape']) if r['shape'] else "N/A",
            r['dtype'] if r['dtype'] else "N/A",
            r['range'] if r['range'] else "N/A"
        )
    
    console.print(table)
    
    # 统计
    total = len(results)
    valid = sum(1 for r in results if r['exists'] and r['shape'] is not None)
    console.print(f"\n[green]✅ 有效特征: {valid}/{total}[/green]")
    
    return results


def check_feature_consistency(mer_results, feat_results):
    """检查MER-Factory输出和CLIP特征的一致性"""
    console.print("\n[bold cyan]🔍 步骤3: 检查一致性[/bold cyan]")
    
    issues = []
    
    for mer, feat in zip(mer_results, feat_results):
        sample = mer['sample']
        
        # 检查：MER有输出但没有特征
        if mer['exists'] and mer['has_summary'] and not feat['exists']:
            issues.append(f"❌ {sample}: 有AU描述但缺少CLIP特征")
        
        # 检查：有特征但没有MER输出
        if feat['exists'] and not (mer['exists'] and mer['has_summary']):
            issues.append(f"⚠️  {sample}: 有CLIP特征但缺少AU描述")
        
        # 检查：描述数量和特征维度不匹配
        if mer['num_descriptions'] > 0 and feat['shape'] is not None:
            if mer['num_descriptions'] != feat['shape'][0]:
                issues.append(f"⚠️  {sample}: 描述数({mer['num_descriptions']}) != 特征数({feat['shape'][0]})")
    
    if issues:
        console.print("[yellow]发现以下问题:[/yellow]")
        for issue in issues:
            console.print(f"  {issue}")
    else:
        console.print("[green]✅ 所有检查通过，数据一致！[/green]")
    
    return len(issues) == 0


def main():
    console.print(Panel.fit(
        "[bold cyan]AU特征提取流程验证[/bold cyan]\n"
        "检查: MER-Factory输出 → CLIP特征提取 → 一致性",
        title="🔍 验证工具"
    ))
    
    # 配置
    mer_factory_root = Path("/home/project/MER-Factory/output/MERCaptionPlus")
    preextracted_root = Path("./preextracted_features/mercaptionplus")
    
    # 获取样本列表（从CSV或取前N个）
    console.print("\n[yellow]正在获取样本列表...[/yellow]")
    
    # 方式1: 从MER-Factory目录获取
    if mer_factory_root.exists():
        sample_dirs = [d.name for d in mer_factory_root.iterdir() if d.is_dir()]
        sample_names = sorted(sample_dirs)[:10]  # 取前10个样本
        console.print(f"[green]✅ 从目录获取 {len(sample_names)} 个样本[/green]")
    else:
        console.print(f"[red]❌ MER-Factory目录不存在: {mer_factory_root}[/red]")
        return
    
    # 显示配置
    console.print("\n[bold]配置信息:[/bold]")
    console.print(f"  MER-Factory: {mer_factory_root}")
    console.print(f"  预提取特征: {preextracted_root}")
    console.print(f"  样本数量: {len(sample_names)}")
    
    # 执行检查
    mer_results = check_mer_factory_output(mer_factory_root, sample_names)
    feat_results = check_extracted_features(preextracted_root, sample_names)
    is_consistent = check_feature_consistency(mer_results, feat_results)
    
    # 总结
    console.print("\n" + "=" * 60)
    if is_consistent:
        console.print("[bold green]🎉 验证通过！可以开始训练。[/bold green]")
        console.print("\n[cyan]下一步:[/cyan]")
        console.print("  1. 确保训练配置中 use_preextracted_features: True")
        console.print("  2. 确保 preextracted_root 路径正确")
        console.print("  3. 运行训练命令")
    else:
        console.print("[bold yellow]⚠️  发现问题，请检查上述警告。[/bold yellow]")
        console.print("\n[cyan]建议:[/cyan]")
        console.print("  1. 重新运行 MER-Factory batch脚本生成AU描述")
        console.print("  2. 重新运行 extract_mercaptionplus_features.sh 提取特征")
    console.print("=" * 60)


if __name__ == '__main__':
    main()
