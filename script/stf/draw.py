#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从log文件中提取epoch为整数的训练数据，并绘制loss随epoch变化的图表
"""

import re
import ast
import os
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 定义log文件所在目录
log_dir = '/home/lcq/data1/_tasks/2510-RDICL_RL/logs/_useful'


def extract_demo_name(filepath):
    """从文件路径中提取demo名称（如demo8, demo16, demo32）"""
    filename = os.path.basename(filepath)
    # 匹配 -demo数字 的模式
    match = re.search(r'-demo(\d+)', filename)
    if match:
        return f"demo{match.group(1)}"
    return "unknown"


def parse_log_file(filepath):
    """解析log文件，提取epoch为整数的行"""
    epochs = []
    losses = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 检查是否是字典格式的行（以{'loss'开头）
            if line.startswith("{'loss'") or line.startswith('{"loss"'):
                try:
                    # 使用ast.literal_eval安全地解析字典
                    data = ast.literal_eval(line)
                    epoch = data.get('epoch')
                    loss = data.get('loss')
                    
                    # 只提取epoch为整数的行（如1.0, 2.0等）
                    if epoch is not None and loss is not None:
                        # 检查epoch是否为整数（浮点数形式的整数，如1.0, 2.0）
                        if epoch == int(epoch):
                            epochs.append(int(epoch))
                            losses.append(loss)
                except (ValueError, SyntaxError):
                    # 如果解析失败，跳过这一行
                    continue
    
    return epochs, losses


def plot_loss_vs_epoch(epochs, losses, demo_name, output_dir):
    """绘制loss随epoch变化的图表"""
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', linewidth=2, markersize=8)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'Loss vs Epoch - {demo_name}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    output_path = os.path.join(output_dir, f'{demo_name}.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存图表: {output_path}")
    plt.close()


def main():
    # 检查目录是否存在
    if not os.path.isdir(log_dir):
        print(f"错误: 目录不存在: {log_dir}")
        return
    
    # 遍历目录下的所有文件，只处理.log文件
    log_files = []
    for filename in os.listdir(log_dir):
        filepath = os.path.join(log_dir, filename)
        # 只处理普通文件且以.log结尾的文件（跳过目录和其他文件）
        if os.path.isfile(filepath) and filename.endswith('.log'):
            log_files.append(filepath)
    
    if not log_files:
        print(f"警告: 目录 {log_dir} 下没有找到文件")
        return
    
    print(f"找到 {len(log_files)} 个文件，开始处理...")
    
    # 处理每个log文件
    for log_file in log_files:
        print(f"\n处理文件: {log_file}")
        
        # 提取demo名称
        demo_name = extract_demo_name(log_file)
        print(f"Demo名称: {demo_name}")
        
        # 获取log文件所在目录作为输出目录
        output_dir = os.path.dirname(log_file)
        
        # 解析log文件
        epochs, losses = parse_log_file(log_file)
        
        if not epochs:
            print(f"警告: 未找到epoch为整数的数据，跳过此文件")
            continue
        
        print(f"找到 {len(epochs)} 个数据点")
        print(f"Epoch范围: {min(epochs)} - {max(epochs)}")
        print(f"Loss范围: {min(losses):.4f} - {max(losses):.4f}")
        
        # 绘制图表
        plot_loss_vs_epoch(epochs, losses, demo_name, output_dir)
    
    print("\n所有图表已生成完成！")


if __name__ == '__main__':
    main()

