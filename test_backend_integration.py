#!/usr/bin/env python3
"""
测试修改后的后端代码是否能正确处理element_extraction.py格式的数据
"""

import sys
import os
sys.path.append('/home/nanqipro01/gitlocal/ai-mouse-analysis/backend')

from src.extraction_logic import get_interactive_data, extract_calcium_features, run_batch_extraction
import pandas as pd

def test_data_reading():
    """测试数据读取功能"""
    print("=== 测试数据读取功能 ===")
    
    # 测试文件路径
    test_file = '/home/nanqipro01/gitlocal/ai-mouse-analysis/dataexample/29790930糖水铁网糖水trace2.xlsx'
    
    if not os.path.exists(test_file):
        print(f"测试文件不存在: {test_file}")
        return False
    
    try:
        # 测试直接读取Excel文件
        df = pd.read_excel(test_file)
        df.columns = [col.strip() for col in df.columns]
        print(f"成功读取数据，形状: {df.shape}")
        
        # 检查神经元列
        neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
        print(f"检测到神经元列数量: {len(neuron_columns)}")
        print(f"前10个神经元列: {neuron_columns[:10]}")
        
        # 检查behavior列
        behavior_cols = [col for col in df.columns if 'behavior' in col.lower()]
        print(f"行为标签列: {behavior_cols}")
        
        return True
        
    except Exception as e:
        print(f"数据读取失败: {e}")
        return False

def test_interactive_data():
    """测试交互式数据获取"""
    print("\n=== 测试交互式数据获取 ===")
    
    test_file = '/home/nanqipro01/gitlocal/ai-mouse-analysis/dataexample/29790930糖水铁网糖水trace2.xlsx'
    
    try:
        # 测试获取交互式数据
        result = get_interactive_data(test_file, 'n4')
        print(f"交互式数据获取成功")
        print(f"数据键: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
        return True
        
    except Exception as e:
        print(f"交互式数据获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_calcium_features():
    """测试钙特征提取"""
    print("\n=== 测试钙特征提取 ===")
    
    test_file = '/home/nanqipro01/gitlocal/ai-mouse-analysis/dataexample/29790930糖水铁网糖水trace2.xlsx'
    
    try:
        # 读取数据
        df = pd.read_excel(test_file)
        df.columns = [col.strip() for col in df.columns]
        
        # 获取第一个神经元的数据
        neuron_columns = [col for col in df.columns if col.startswith('n') and col[1:].isdigit()]
        if not neuron_columns:
            print("没有找到神经元列")
            return False
            
        neuron_data = df[neuron_columns[0]].values
        print(f"测试神经元: {neuron_columns[0]}，数据长度: {len(neuron_data)}")
        
        # 提取特征
        feature_table, fig, smoothed_data = extract_calcium_features(
            neuron_data, fs=4.8, visualize=False
        )
        
        print(f"特征提取成功")
        if not feature_table.empty:
            print(f"检测到 {len(feature_table)} 个钙事件")
            print(f"特征列: {list(feature_table.columns)}")
        else:
            print("未检测到钙事件")
            
        return True
        
    except Exception as e:
        print(f"钙特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始测试修改后的后端代码...\n")
    
    tests = [
        ("数据读取", test_data_reading),
        ("交互式数据获取", test_interactive_data),
        ("钙特征提取", test_calcium_features)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"{test_name} 测试出现异常: {e}")
            results.append((test_name, False))
    
    print("\n=== 测试结果汇总 ===")
    all_passed = True
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        print("\n🎉 所有测试通过！后端代码已成功适配element_extraction.py格式")
    else:
        print("\n❌ 部分测试失败，需要进一步调试")
    
    return all_passed

if __name__ == "__main__":
    main()