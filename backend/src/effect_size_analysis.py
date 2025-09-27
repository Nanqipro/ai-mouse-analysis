"""
效应量分析模块
集成effect_size-1模块的功能，提供效应量计算和分析服务
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
from pathlib import Path


class EffectSizeCalculator:
    """
    效应量计算器类：用于计算神经元活动与行为之间的Cohen's d效应量
    """
    
    def __init__(self, behavior_labels: List[str] = None):
        """
        初始化效应量计算器
        
        参数
        ----------
        behavior_labels : List[str], 可选
            行为标签列表
        """
        self.behavior_labels = behavior_labels
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.nan_info = {}  # 存储NaN值处理信息
        
    def remove_nan_rows(self, neuron_data: np.ndarray, behavior_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        移除包含NaN值的行
        
        参数
        ----------
        neuron_data : np.ndarray
            神经元数据
        behavior_data : np.ndarray
            行为数据
            
        返回
        ----------
        Tuple[np.ndarray, np.ndarray]
            清理后的神经元数据和行为数据
        """
        # 确保神经元数据是数值类型
        try:
            neuron_data = pd.DataFrame(neuron_data).apply(pd.to_numeric, errors='coerce').values
        except Exception as e:
            print(f"警告: 神经元数据转换失败: {e}")
        
        # 检查神经元数据中的NaN值
        try:
            neuron_nan_mask = np.isnan(neuron_data).any(axis=1)
        except Exception as e:
            print(f"警告: 神经元数据NaN检查失败: {e}")
            neuron_nan_mask = np.zeros(neuron_data.shape[0], dtype=bool)
        
        # 检查行为数据中的NaN值
        try:
            if behavior_data.dtype.kind in ['U', 'S', 'O']:  # 字符串或对象类型
                behavior_nan_mask = pd.isna(pd.Series(behavior_data))
            else:
                behavior_nan_mask = np.isnan(behavior_data)
        except Exception as e:
            print(f"警告: 行为数据NaN检查失败: {e}")
            # 创建一个通用的检测方法
            behavior_nan_list = []
            for item in behavior_data:
                if item is None or item is np.nan:
                    behavior_nan_list.append(True)
                elif isinstance(item, str) and (item.lower() in ['nan', 'none', ''] or item.strip() == ''):
                    behavior_nan_list.append(True)
                else:
                    behavior_nan_list.append(False)
            behavior_nan_mask = np.array(behavior_nan_list)
        
        # 合并NaN掩码
        total_nan_mask = neuron_nan_mask | behavior_nan_mask
        
        # 记录NaN信息
        self.nan_info = {
            'total_rows': len(neuron_data),
            'nan_rows': total_nan_mask.sum(),
            'neuron_nan_rows': neuron_nan_mask.sum(),
            'behavior_nan_rows': behavior_nan_mask.sum()
        }
        
        if total_nan_mask.sum() > 0:
            print(f"发现 {total_nan_mask.sum()} 行包含NaN值，将被移除")
            print(f"神经元数据NaN行数: {neuron_nan_mask.sum()}")
            print(f"行为数据NaN行数: {behavior_nan_mask.sum()}")
        
        # 移除NaN行
        clean_neuron_data = neuron_data[~total_nan_mask]
        clean_behavior_data = behavior_data[~total_nan_mask]
        
        return clean_neuron_data, clean_behavior_data
    
    def preprocess_data(self, neuron_data: np.ndarray, behavior_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        数据预处理：移除NaN值、标准化神经元数据、编码行为标签
        
        参数
        ----------
        neuron_data : np.ndarray
            原始神经元活动数据
        behavior_data : np.ndarray
            原始行为标签数据
            
        返回
        ----------
        Tuple[np.ndarray, np.ndarray]
            标准化后的神经元数据和编码后的行为标签
        """
        print("开始数据预处理...")
        
        # 移除NaN值
        neuron_data, behavior_data = self.remove_nan_rows(neuron_data, behavior_data)
        
        # 标准化神经元数据
        print("标准化神经元数据...")
        X_scaled = self.scaler.fit_transform(neuron_data)
        
        # 编码行为标签
        print("编码行为标签...")
        y_encoded = self.label_encoder.fit_transform(behavior_data)
        
        # 获取行为标签列表
        self.behavior_labels = self.label_encoder.classes_
        print(f"发现行为标签: {self.behavior_labels}")
        
        print(f"预处理完成: 神经元数据 {X_scaled.shape}, 行为数据 {y_encoded.shape}")
        
        return X_scaled, y_encoded
    
    def calculate_effect_sizes(self, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算Cohen's d效应量
        
        参数
        ----------
        X_scaled : np.ndarray
            标准化后的神经元数据
        y : np.ndarray
            编码后的行为标签
            
        返回
        ----------
        Dict[str, np.ndarray]
            每种行为的效应量数组
        """
        print("\n计算Cohen's d效应量...")
        
        effect_sizes = {}
        
        for behavior_idx, behavior in enumerate(self.behavior_labels):
            print(f"计算行为 '{behavior}' 的效应量...")
            
            # 分离该行为和其他行为的数据
            behavior_mask = (y == behavior_idx)
            behavior_data = X_scaled[behavior_mask]
            other_data = X_scaled[~behavior_mask]
            
            if len(behavior_data) == 0:
                print(f"警告: 行为 '{behavior}' 没有样本数据")
                effect_sizes[behavior] = np.zeros(X_scaled.shape[1])
                continue
                
            if len(other_data) == 0:
                print(f"警告: 除行为 '{behavior}' 外没有其他样本数据")
                effect_sizes[behavior] = np.zeros(X_scaled.shape[1])
                continue
            
            # 计算均值和标准差
            behavior_mean = np.nanmean(behavior_data, axis=0)
            other_mean = np.nanmean(other_data, axis=0)
            behavior_std = np.nanstd(behavior_data, axis=0, ddof=1)
            other_std = np.nanstd(other_data, axis=0, ddof=1)
            
            # 计算合并标准差
            pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
            
            # 处理NaN和零值情况
            nan_mask = np.isnan(pooled_std)
            if nan_mask.any():
                print(f"警告: 行为 '{behavior}' 中有 {nan_mask.sum()} 个神经元的标准差为NaN")
                pooled_std[nan_mask] = np.maximum(behavior_std[nan_mask], other_std[nan_mask])
            
            # 避免除零错误
            zero_mask = (pooled_std == 0)
            if zero_mask.any():
                print(f"警告: 行为 '{behavior}' 中有 {zero_mask.sum()} 个神经元的标准差为零")
                pooled_std[zero_mask] = 1e-10  # 使用很小的值避免除零
            
            # 计算Cohen's d
            cohens_d = (behavior_mean - other_mean) / pooled_std
            
            # 处理NaN结果
            if np.isnan(cohens_d).any():
                print(f"警告: 行为 '{behavior}' 的效应量计算中包含NaN值")
                cohens_d = np.nan_to_num(cohens_d, nan=0.0)
            
            effect_sizes[behavior] = cohens_d
            
            # 统计信息
            significant_neurons = np.sum(np.abs(cohens_d) > 0.2)  # 小效应量阈值
            print(f"行为 '{behavior}': {significant_neurons} 个神经元具有显著效应量")
        
        return effect_sizes
    
    def get_top_neurons_per_behavior(self, effect_sizes: Dict[str, np.ndarray], 
                                   top_n: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        获取每种行为中效应量最大的前N个神经元
        
        参数
        ----------
        effect_sizes : Dict[str, np.ndarray]
            效应量字典
        top_n : int
            返回的神经元数量
            
        返回
        ----------
        Dict[str, Dict[str, Any]]
            每种行为的top神经元信息
        """
        top_neurons = {}
        
        for behavior, effects in effect_sizes.items():
            # 获取绝对值最大的神经元索引
            abs_effects = np.abs(effects)
            top_indices = np.argsort(abs_effects)[-top_n:][::-1]
            
            top_neurons[behavior] = {
                'indices': top_indices.tolist(),
                'effect_sizes': effects[top_indices].tolist(),
                'abs_effect_sizes': abs_effects[top_indices].tolist(),
                'neuron_ids': [f"Neuron_{i+1}" for i in top_indices]
            }
        
        return top_neurons
    
    def export_effect_sizes_to_csv(self, effect_sizes: Dict[str, np.ndarray], 
                                 output_path: str) -> None:
        """
        将效应量结果导出为CSV文件
        
        参数
        ----------
        effect_sizes : Dict[str, np.ndarray]
            效应量字典
        output_path : str
            输出文件路径
        """
        # 创建DataFrame
        df = pd.DataFrame(effect_sizes)
        
        # 添加神经元ID列
        df.index = [f"Neuron_{i+1}" for i in range(len(df))]
        df.index.name = "Neuron_ID"
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存文件
        df.to_csv(output_path)
        print(f"效应量结果已保存到: {output_path}")
    
    def calculate_effect_sizes_from_raw_data(self, neuron_data: np.ndarray, 
                                           behavior_data: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        从原始数据计算效应量的完整流程
        
        参数
        ----------
        neuron_data : np.ndarray
            原始神经元活动数据
        behavior_data : np.ndarray
            原始行为标签数据
            
        返回
        ----------
        effect_sizes : Dict[str, np.ndarray]
            每种行为的效应量数组
        X_scaled : np.ndarray
            标准化后的神经元数据
        y_encoded : np.ndarray
            编码后的行为标签
        """
        print("开始从原始数据计算效应量的完整流程...")
        
        # 预处理数据
        X_scaled, y_encoded = self.preprocess_data(neuron_data, behavior_data)
        
        # 计算效应量
        effect_sizes = self.calculate_effect_sizes(X_scaled, y_encoded)
        
        print("效应量计算完整流程完成！")
        
        return effect_sizes, X_scaled, y_encoded


def analyze_effect_sizes(data: pd.DataFrame, behavior_column: str = None) -> Dict[str, Any]:
    """
    分析效应量的主函数
    
    参数
    ----------
    data : pd.DataFrame
        输入数据
    behavior_column : str, 可选
        行为标签列名，如果为None则使用最后一列
        
    返回
    ----------
    Dict[str, Any]
        分析结果
    """
    print("开始效应量分析...")
    
    # 检查数据中的NaN值
    total_nan = data.isnull().sum().sum()
    if total_nan > 0:
        print(f"数据中发现 {total_nan} 个NaN值")
        nan_cols = data.columns[data.isnull().any()].tolist()
        print(f"包含NaN值的列: {nan_cols}")
    else:
        print("数据中没有发现NaN值")
    
    # 分离神经元数据和行为标签
    if behavior_column is None:
        # 使用最后一列作为行为标签
        neuron_df = data.iloc[:, :-1]
        behavior_data = data.iloc[:, -1].values
        print(f"使用最后一列 '{data.columns[-1]}' 作为行为标签")
    else:
        if behavior_column not in data.columns:
            raise ValueError(f"指定的行为标签列 '{behavior_column}' 不存在")
        neuron_df = data.drop(columns=[behavior_column])
        behavior_data = data[behavior_column].values
        print(f"使用列 '{behavior_column}' 作为行为标签")
    
    # 确保神经元数据是数值类型
    print("转换神经元数据为数值类型...")
    try:
        # 尝试将所有神经元列转换为数值类型
        neuron_df = neuron_df.apply(pd.to_numeric, errors='coerce')
        neuron_data = neuron_df.values
        print(f"神经元数据转换成功: {neuron_data.shape}")
    except Exception as e:
        print(f"警告: 神经元数据转换失败: {e}")
        # 如果转换失败，尝试逐列转换
        for col in neuron_df.columns:
            try:
                neuron_df[col] = pd.to_numeric(neuron_df[col], errors='coerce')
            except Exception as col_e:
                print(f"警告: 列 '{col}' 转换失败: {col_e}")
        neuron_data = neuron_df.values
    
    print(f"神经元数据: {neuron_data.shape}, 行为数据: {behavior_data.shape}")
    
    # 创建效应量计算器并计算
    calculator = EffectSizeCalculator()
    effect_sizes, X_scaled, y_encoded = calculator.calculate_effect_sizes_from_raw_data(
        neuron_data, behavior_data
    )
    
    # 获取top神经元
    top_neurons = calculator.get_top_neurons_per_behavior(effect_sizes, top_n=10)
    
    # 整理结果
    results = {
        'effect_sizes': effect_sizes,
        'behavior_labels': calculator.behavior_labels.tolist(),
        'top_neurons': top_neurons,
        'nan_info': calculator.nan_info,
        'processed_data': {
            'X_scaled': X_scaled,
            'y_encoded': y_encoded
        },
        'data_summary': {
            'total_neurons': neuron_data.shape[1],
            'total_samples': neuron_data.shape[0],
            'behavior_counts': dict(zip(*np.unique(behavior_data, return_counts=True)))
        }
    }
    
    print(f"\n效应量分析完成！")
    print(f"分析了 {neuron_data.shape[1]} 个神经元，{neuron_data.shape[0]} 个样本")
    print(f"发现 {len(calculator.behavior_labels)} 种行为: {calculator.behavior_labels}")
    
    return results
