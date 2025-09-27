# -*- coding: utf-8 -*-
"""
Matplotlib 全局配置文件

该文件用于统一设置项目中所有图表的字体大小和样式，
确保图表的坐标轴标签、标题等文字清晰可读。

使用方法：
在需要绘图的模块开头导入此配置：
from matplotlib_config import setup_matplotlib_style
setup_matplotlib_style()

作者：Assistant
日期：2025年
"""

import matplotlib.pyplot as plt
import matplotlib as mpl

def setup_matplotlib_style():
    """
    设置matplotlib的全局样式，包括字体大小和粗细
    
    该函数会设置以下样式：
    - 坐标轴标签字体大小和粗细
    - 刻度标签字体大小和粗细
    - 图表标题字体大小和粗细
    - 图例字体大小
    - 中文字体支持
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 设置坐标轴标签字体大小和粗细 - 适合论文使用的大字体
    plt.rcParams['axes.labelsize'] = 20  # 坐标轴标签字体大小 (从14增加到20)
    plt.rcParams['axes.labelweight'] = 'bold'  # 坐标轴标签字体粗细
    
    # 设置刻度标签字体大小和粗细
    plt.rcParams['xtick.labelsize'] = 18  # X轴刻度字体大小 (从12增加到18)
    plt.rcParams['ytick.labelsize'] = 18  # Y轴刻度字体大小 (从12增加到18)
    plt.rcParams['xtick.major.size'] = 8  # X轴主刻度长度 (从6增加到8)
    plt.rcParams['ytick.major.size'] = 8  # Y轴主刻度长度 (从6增加到8)
    plt.rcParams['xtick.minor.size'] = 6  # X轴次刻度长度 (从4增加到6)
    plt.rcParams['ytick.minor.size'] = 6  # Y轴次刻度长度 (从4增加到6)
    
    # 设置图表标题字体大小和粗细
    plt.rcParams['axes.titlesize'] = 24  # 子图标题字体大小 (从16增加到24)
    plt.rcParams['axes.titleweight'] = 'bold'  # 子图标题字体粗细
    plt.rcParams['figure.titlesize'] = 26  # 主图标题字体大小 (从18增加到26)
    plt.rcParams['figure.titleweight'] = 'bold'  # 主图标题字体粗细
    
    # 设置图例字体大小
    plt.rcParams['legend.fontsize'] = 18  # 图例字体大小 (调整为18)
    plt.rcParams['legend.markerscale'] = 1.5  # 图例标记大小比例
    plt.rcParams['legend.handlelength'] = 2.0  # 图例标记长度
    plt.rcParams['legend.handletextpad'] = 0.8  # 图例标记与文本间距
    
    # 设置图表边框和网格
    plt.rcParams['axes.linewidth'] = 1.2  # 坐标轴边框线宽
    plt.rcParams['grid.linewidth'] = 0.8  # 网格线宽
    plt.rcParams['grid.alpha'] = 0.5  # 网格透明度
    
    # 设置图表保存参数 - 适合论文发表的高质量设置
    plt.rcParams['savefig.dpi'] = 600  # 保存图片的DPI (从300增加到600)
    plt.rcParams['savefig.bbox'] = 'tight'  # 保存时自动调整边界
    plt.rcParams['savefig.facecolor'] = 'white'  # 保存图片的背景色
    plt.rcParams['savefig.edgecolor'] = 'none'  # 保存图片的边框色
    plt.rcParams['figure.figsize'] = [10, 10]  # 默认图片尺寸 (英寸) - 1:1正方形
    
    print("✅ Matplotlib样式配置已应用：字体大小和粗细已优化")

def reset_matplotlib_style():
    """
    重置matplotlib样式为默认设置
    """
    mpl.rcdefaults()
    print("✅ Matplotlib样式已重置为默认设置")

def get_font_config():
    """
    获取当前字体配置信息
    
    Returns:
        dict: 包含当前字体配置的字典
    """
    return {
        'axes_labelsize': plt.rcParams['axes.labelsize'],
        'axes_labelweight': plt.rcParams['axes.labelweight'],
        'xtick_labelsize': plt.rcParams['xtick.labelsize'],
        'ytick_labelsize': plt.rcParams['ytick.labelsize'],
        'axes_titlesize': plt.rcParams['axes.titlesize'],
        'axes_titleweight': plt.rcParams['axes.titleweight'],
        'figure_titlesize': plt.rcParams['figure.titlesize'],
        'legend_fontsize': plt.rcParams['legend.fontsize']
    }

# 自动应用样式（当模块被导入时）
if __name__ != '__main__':
    setup_matplotlib_style()

if __name__ == '__main__':
    # 测试配置
    print("=== Matplotlib配置测试 ===")
    setup_matplotlib_style()
    
    # 显示当前配置
    config = get_font_config()
    print("\n当前字体配置：")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建测试图表
    import numpy as np
    
    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    
    ax.plot(x, y, label='sin(x)')
    ax.set_xlabel('X轴标签 (测试字体)')
    ax.set_ylabel('Y轴标签 (测试字体)')
    ax.set_title('测试图表标题 (字体大小和粗细)')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('matplotlib_config_test.png')
    print("\n✅ 测试图表已保存为 'matplotlib_config_test.png'")
    plt.show()