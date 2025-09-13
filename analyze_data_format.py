import pandas as pd
import os

# 分析示例数据文件的格式
data_file = '/home/nanqipro01/gitlocal/ai-mouse-analysis/dataexample/29790930糖水铁网糖水trace2.xlsx'

if os.path.exists(data_file):
    try:
        # 读取Excel文件，查看所有工作表
        excel_file = pd.ExcelFile(data_file)
        print(f"工作表列表: {excel_file.sheet_names}")
        
        # 读取每个工作表的前几行
        for sheet_name in excel_file.sheet_names:
            print(f"\n=== 工作表: {sheet_name} ===")
            df = pd.read_excel(data_file, sheet_name=sheet_name)
            print(f"数据形状: {df.shape}")
            print(f"列名: {list(df.columns)}")
            print("前5行数据:")
            print(df.head())
            
            # 检查神经元列
            neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
            print(f"检测到的神经元列: {neuron_columns[:10]}...")  # 只显示前10个
            print(f"神经元列总数: {len(neuron_columns)}")
            
            # 检查是否有行为标签列
            behavior_cols = [col for col in df.columns if 'behavior' in col.lower()]
            print(f"行为标签列: {behavior_cols}")
            
    except Exception as e:
        print(f"读取文件时出错: {e}")
else:
    print(f"文件不存在: {data_file}")