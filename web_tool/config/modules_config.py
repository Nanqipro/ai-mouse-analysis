#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法模块配置文件

定义了所有可用的算法模块及其参数配置
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALGORITHM_DIR = os.path.join(os.path.dirname(BASE_DIR), 'algorithm')
DATA_EXAMPLE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'dataexample')
UPLOADS_DIR = os.path.join(BASE_DIR, 'uploads')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# 算法模块配置
MODULES_CONFIG = {
    'cluster': {
        'name': '钙爆发事件聚类分析',
        'description': '对神经元钙爆发事件进行聚类分析，识别不同的活动模式',
        'script_path': os.path.join(ALGORITHM_DIR, 'cluster', 'cluster.py'),
        'input_format': 'xlsx',
        'required_columns': ['neuron', 'start_time', 'end_time', 'amplitude', 'duration'],
        'parameters': {
            'k': {
                'type': 'int',
                'default': None,
                'min': 2,
                'max': 20,
                'description': '聚类数K，不指定则自动确定最佳值',
                'required': False
            },
            'compare': {
                'type': 'str',
                'default': None,
                'description': '比较多个K值的效果，格式如"2,3,4,5"',
                'required': False
            }
        },
        'outputs': [
            {'type': 'image', 'name': '聚类结果可视化', 'format': 'png'},
            {'type': 'data', 'name': '聚类结果数据', 'format': 'xlsx'},
            {'type': 'report', 'name': '聚类分析报告', 'format': 'txt'}
        ]
    },
    
    'heatmap': {
        'name': '行为热力图分析',
        'description': '生成特定行为前后时间窗口的神经元活动热力图',
        'script_path': os.path.join(ALGORITHM_DIR, 'heatmap', 'heatmap_behavior.py'),
        'input_format': 'xlsx',
        'required_columns': ['stamp', 'behavior', 'neuron_columns'],
        'parameters': {
            'start_behavior': {
                'type': 'str',
                'default': 'Crack-seeds-shells',
                'description': '起始行为类型',
                'required': True,
                'options': ['Crack-seeds-shells', 'Eat-feed', 'Eat-seed-kernels', 'Explore', 
                           'Groom', 'Water', 'Close-arm', 'Middle-zone', 'Open-arm']
            },
            'end_behavior': {
                'type': 'str',
                'default': 'Eat-seed-kernels',
                'description': '结束行为类型',
                'required': True,
                'options': ['Crack-seeds-shells', 'Eat-feed', 'Eat-seed-kernels', 'Explore', 
                           'Groom', 'Water', 'Close-arm', 'Middle-zone', 'Open-arm']
            },
            'pre_behavior_time': {
                'type': 'float',
                'default': 5.0,
                'min': 0.0,
                'max': 30.0,
                'description': '行为开始前的时间（秒）',
                'required': True
            },
            'sorting_method': {
                'type': 'str',
                'default': 'global',
                'description': '神经元排序方式',
                'required': True,
                'options': ['global', 'local', 'first', 'custom']
            }
        },
        'outputs': [
            {'type': 'image', 'name': '行为热力图', 'format': 'png'},
            {'type': 'image', 'name': '平均热力图', 'format': 'png'},
            {'type': 'data', 'name': '热力图数据', 'format': 'csv'}
        ]
    },
    
    'trace': {
        'name': '神经元活动轨迹图',
        'description': '绘制神经元活动的时间序列轨迹图',
        'script_path': os.path.join(ALGORITHM_DIR, 'trace', 'show_trace.py'),
        'input_format': 'xlsx',
        'required_columns': ['stamp', 'neuron_columns', 'behavior'],
        'parameters': {
            'stamp_min': {
                'type': 'float',
                'default': None,
                'description': '最小时间戳值',
                'required': False
            },
            'stamp_max': {
                'type': 'float',
                'default': None,
                'description': '最大时间戳值',
                'required': False
            },
            'sort_method': {
                'type': 'str',
                'default': 'peak',
                'description': '排序方式',
                'required': True,
                'options': ['original', 'peak', 'calcium_wave', 'custom']
            },
            'max_neurons': {
                'type': 'int',
                'default': 60,
                'min': 10,
                'max': 200,
                'description': '最大显示神经元数量',
                'required': True
            },
            'scaling_factor': {
                'type': 'float',
                'default': 80.0,
                'min': 10.0,
                'max': 200.0,
                'description': '信号振幅缩放因子',
                'required': True
            },
            'line_width': {
                'type': 'float',
                'default': 2.0,
                'min': 0.5,
                'max': 5.0,
                'description': 'trace线的宽度',
                'required': True
            }
        },
        'outputs': [
            {'type': 'image', 'name': '神经元轨迹图', 'format': 'png'}
        ]
    },
    
    'effect_size': {
        'name': '效应量计算',
        'description': '计算神经元在不同行为状态下的效应量',
        'script_path': os.path.join(ALGORITHM_DIR, 'effect_size', 'effect_size_calculator.py'),
        'input_format': 'xlsx',
        'required_columns': ['behavior', 'neuron_columns'],
        'parameters': {
            'behavior_column': {
                'type': 'str',
                'default': 'behavior',
                'description': '行为标签列名',
                'required': True
            },
            'threshold': {
                'type': 'float',
                'default': 0.4,
                'min': 0.1,
                'max': 1.0,
                'description': '效应量阈值',
                'required': True
            }
        },
        'outputs': [
            {'type': 'data', 'name': '效应量数据', 'format': 'csv'},
            {'type': 'image', 'name': '效应量分布图', 'format': 'png'},
            {'type': 'image', 'name': '效应量箱线图', 'format': 'png'}
        ]
    },
    
    'principal_neuron': {
        'name': '关键神经元分析',
        'description': '识别和分析在特定行为状态下起关键作用的神经元',
        'script_path': os.path.join(ALGORITHM_DIR, 'principal_neuron', 'src', 'main_emtrace01_analysis.py'),
        'input_format': 'xlsx',
        'required_columns': ['behavior', 'neuron_columns'],
        'additional_files': {
            'position_file': {
                'description': '神经元位置数据文件',
                'format': 'csv',
                'required_columns': ['NeuronID', 'X', 'Y'],
                'required': True
            }
        },
        'parameters': {
            'effect_size_threshold': {
                'type': 'float',
                'default': 0.4407,
                'min': 0.1,
                'max': 1.0,
                'description': '效应量阈值',
                'required': True
            },
            'recalculate': {
                'type': 'bool',
                'default': False,
                'description': '是否重新计算效应量',
                'required': False
            },
            'show_background_neurons': {
                'type': 'bool',
                'default': True,
                'description': '是否显示背景神经元',
                'required': False
            }
        },
        'outputs': [
            {'type': 'image', 'name': '关键神经元分布图', 'format': 'png'},
            {'type': 'image', 'name': '共享神经元图', 'format': 'png'},
            {'type': 'image', 'name': '独有神经元图', 'format': 'png'},
            {'type': 'data', 'name': '效应量数据', 'format': 'csv'},
            {'type': 'report', 'name': '分析报告', 'format': 'txt'}
        ]
    }
}

# 支持的文件格式
SUPPORTED_FORMATS = {
    'xlsx': 'Excel文件',
    'csv': 'CSV文件',
    'txt': '文本文件'
}

# 输出文件类型
OUTPUT_TYPES = {
    'image': '图像文件',
    'data': '数据文件',
    'report': '报告文件'
}

# 示例数据文件
EXAMPLE_DATA = {
    'main_data': os.path.join(DATA_EXAMPLE_DIR, '29790930糖水铁网糖水trace2.xlsx'),
    'description': '示例神经元活动数据，包含时间戳、行为标签和神经元活动数据'
}

def get_module_config(module_name):
    """获取指定模块的配置"""
    return MODULES_CONFIG.get(module_name)

def get_all_modules():
    """获取所有可用模块"""
    return list(MODULES_CONFIG.keys())

def validate_parameters(module_name, parameters):
    """验证模块参数"""
    config = get_module_config(module_name)
    if not config:
        return False, f"未知模块: {module_name}"
    
    errors = []
    for param_name, param_config in config['parameters'].items():
        if param_config.get('required', False) and param_name not in parameters:
            errors.append(f"缺少必需参数: {param_name}")
        
        if param_name in parameters:
            value = parameters[param_name]
            param_type = param_config['type']
            
            # 类型检查
            if param_type == 'int' and not isinstance(value, int):
                try:
                    value = int(value)
                    parameters[param_name] = value
                except ValueError:
                    errors.append(f"参数 {param_name} 必须是整数")
            
            elif param_type == 'float' and not isinstance(value, (int, float)):
                try:
                    value = float(value)
                    parameters[param_name] = value
                except ValueError:
                    errors.append(f"参数 {param_name} 必须是数字")
            
            elif param_type == 'bool' and not isinstance(value, bool):
                if isinstance(value, str):
                    value = value.lower() in ['true', '1', 'yes', 'on']
                    parameters[param_name] = value
                else:
                    errors.append(f"参数 {param_name} 必须是布尔值")
            
            # 范围检查
            if param_type in ['int', 'float']:
                if 'min' in param_config and value < param_config['min']:
                    errors.append(f"参数 {param_name} 不能小于 {param_config['min']}")
                if 'max' in param_config and value > param_config['max']:
                    errors.append(f"参数 {param_name} 不能大于 {param_config['max']}")
            
            # 选项检查
            if 'options' in param_config and value not in param_config['options']:
                errors.append(f"参数 {param_name} 必须是以下选项之一: {param_config['options']}")
    
    return len(errors) == 0, errors