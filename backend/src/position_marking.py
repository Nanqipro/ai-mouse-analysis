"""
位置标记模块
集成getposition-1模块的功能，提供神经元位置标记服务
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import os
import json
from pathlib import Path


class PositionMarker:
    """
    位置标记器类：用于处理神经元位置标记数据
    """
    
    def __init__(self):
        """初始化位置标记器"""
        self.marked_points = {}  # 存储标记的点
        self.current_number = 1  # 当前编号
        
    def add_point(self, x: float, y: float, neuron_id: int = None) -> int:
        """
        添加一个标记点
        
        参数
        ----------
        x : float
            X坐标（相对坐标，0-1范围）
        y : float
            Y坐标（相对坐标，0-1范围）
        neuron_id : int, 可选
            神经元ID，如果为None则自动分配
            
        返回
        ----------
        int
            神经元ID
        """
        if neuron_id is None:
            neuron_id = self.current_number
            self.current_number += 1
        
        self.marked_points[neuron_id] = {
            'x': x,
            'y': y,
            'relative_x': x,
            'relative_y': y
        }
        
        return neuron_id
    
    def remove_point(self, neuron_id: int) -> bool:
        """
        移除一个标记点
        
        参数
        ----------
        neuron_id : int
            神经元ID
            
        返回
        ----------
        bool
            是否成功移除
        """
        if neuron_id in self.marked_points:
            del self.marked_points[neuron_id]
            return True
        return False
    
    def update_point(self, neuron_id: int, x: float, y: float) -> bool:
        """
        更新一个标记点的位置
        
        参数
        ----------
        neuron_id : int
            神经元ID
        x : float
            新的X坐标
        y : float
            新的Y坐标
            
        返回
        ----------
        bool
            是否成功更新
        """
        if neuron_id in self.marked_points:
            self.marked_points[neuron_id]['x'] = x
            self.marked_points[neuron_id]['y'] = y
            self.marked_points[neuron_id]['relative_x'] = x
            self.marked_points[neuron_id]['relative_y'] = y
            return True
        return False
    
    def get_all_points(self) -> Dict[int, Dict[str, float]]:
        """
        获取所有标记点
        
        返回
        ----------
        Dict[int, Dict[str, float]]
            所有标记点的字典
        """
        return self.marked_points.copy()
    
    def export_to_csv(self, output_path: str) -> None:
        """
        导出标记点到CSV文件
        
        参数
        ----------
        output_path : str
            输出文件路径
        """
        if not self.marked_points:
            raise ValueError("没有标记任何点")
        
        # 创建DataFrame
        data = []
        for neuron_id, coords in self.marked_points.items():
            data.append({
                'number': neuron_id,
                'relative_x': coords['relative_x'],
                'relative_y': coords['relative_y']
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('number')  # 按编号排序
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存文件
        df.to_csv(output_path, index=False)
        print(f"位置数据已保存到: {output_path}")
    
    def load_from_csv(self, input_path: str) -> None:
        """
        从CSV文件加载标记点
        
        参数
        ----------
        input_path : str
            输入文件路径
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"文件不存在: {input_path}")
        
        df = pd.read_csv(input_path)
        
        # 清空现有数据
        self.marked_points = {}
        
        # 加载数据
        for _, row in df.iterrows():
            neuron_id = int(row['number'])
            self.marked_points[neuron_id] = {
                'x': row['relative_x'],
                'y': row['relative_y'],
                'relative_x': row['relative_x'],
                'relative_y': row['relative_y']
            }
        
        # 更新当前编号
        if self.marked_points:
            self.current_number = max(self.marked_points.keys()) + 1
        else:
            self.current_number = 1
        
        print(f"从 {input_path} 加载了 {len(self.marked_points)} 个标记点")
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取标记统计信息
        
        返回
        ----------
        Dict[str, Any]
            统计信息
        """
        if not self.marked_points:
            return {
                'total_points': 0,
                'min_x': 0,
                'max_x': 0,
                'min_y': 0,
                'max_y': 0,
                'coverage': 0
            }
        
        x_coords = [coords['x'] for coords in self.marked_points.values()]
        y_coords = [coords['y'] for coords in self.marked_points.values()]
        
        return {
            'total_points': len(self.marked_points),
            'min_x': min(x_coords),
            'max_x': max(x_coords),
            'min_y': min(y_coords),
            'max_y': max(y_coords),
            'coverage': (max(x_coords) - min(x_coords)) * (max(y_coords) - min(y_coords))
        }


def process_position_data(points_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    处理位置标记数据
    
    参数
    ----------
    points_data : List[Dict[str, Any]]
        标记点数据列表
        
    返回
    ----------
    Dict[str, Any]
        处理结果
    """
    marker = PositionMarker()
    
    # 添加所有点
    for point in points_data:
        marker.add_point(
            x=point['x'],
            y=point['y'],
            neuron_id=point.get('neuron_id')
        )
    
    # 获取统计信息
    stats = marker.get_statistics()
    
    return {
        'success': True,
        'total_points': stats['total_points'],
        'statistics': stats,
        'points': marker.get_all_points()
    }


def validate_position_data(position_data: pd.DataFrame) -> Dict[str, Any]:
    """
    验证位置数据的格式和完整性
    
    参数
    ----------
    position_data : pd.DataFrame
        位置数据
        
    返回
    ----------
    Dict[str, Any]
        验证结果
    """
    errors = []
    warnings = []
    
    # 检查必需的列
    required_columns = ['number', 'relative_x', 'relative_y']
    missing_columns = [col for col in required_columns if col not in position_data.columns]
    
    if missing_columns:
        errors.append(f"缺少必需的列: {missing_columns}")
        return {
            'valid': False,
            'errors': errors,
            'warnings': warnings
        }
    
    # 检查数据类型
    if not pd.api.types.is_numeric_dtype(position_data['number']):
        errors.append("'number' 列必须是数值类型")
    
    if not pd.api.types.is_numeric_dtype(position_data['relative_x']):
        errors.append("'relative_x' 列必须是数值类型")
    
    if not pd.api.types.is_numeric_dtype(position_data['relative_y']):
        errors.append("'relative_y' 列必须是数值类型")
    
    # 检查坐标范围
    if position_data['relative_x'].min() < 0 or position_data['relative_x'].max() > 1:
        warnings.append("'relative_x' 值超出 [0, 1] 范围")
    
    if position_data['relative_y'].min() < 0 or position_data['relative_y'].max() > 1:
        warnings.append("'relative_y' 值超出 [0, 1] 范围")
    
    # 检查重复的神经元编号
    duplicate_numbers = position_data['number'].duplicated().sum()
    if duplicate_numbers > 0:
        errors.append(f"发现 {duplicate_numbers} 个重复的神经元编号")
    
    # 检查NaN值
    nan_counts = position_data[required_columns].isnull().sum()
    if nan_counts.any():
        errors.append(f"发现NaN值: {nan_counts.to_dict()}")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'total_points': len(position_data),
        'coordinate_range': {
            'x_min': position_data['relative_x'].min(),
            'x_max': position_data['relative_x'].max(),
            'y_min': position_data['relative_y'].min(),
            'y_max': position_data['relative_y'].max()
        }
    }

