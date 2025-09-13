#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
算法模块适配器

为各个算法模块提供统一的调用接口
"""

import os
import sys
import subprocess
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional

# 添加算法模块路径
ALGORITHM_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'algorithm')
sys.path.append(ALGORITHM_DIR)

class ModuleAdapter(ABC):
    """算法模块适配器基类"""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
    
    @abstractmethod
    def validate_input(self, file_path: str, additional_files: Dict[str, str] = None) -> tuple:
        """验证输入文件"""
        pass
    
    @abstractmethod
    def run_analysis(self, file_path: str, output_dir: str, parameters: Dict[str, Any], 
                    additional_files: Dict[str, str] = None) -> Dict[str, Any]:
        """运行分析"""
        pass
    
    def prepare_output_dir(self, output_dir: str):
        """准备输出目录"""
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

class ClusterAdapter(ModuleAdapter):
    """聚类分析模块适配器"""
    
    def __init__(self):
        super().__init__('cluster')
    
    def validate_input(self, file_path: str, additional_files: Dict[str, str] = None) -> tuple:
        try:
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            
            required_cols = ['neuron', 'start_time', 'end_time', 'amplitude', 'duration']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                return False, f"缺少必需的列: {', '.join(missing_cols)}"
            
            return True, f"验证成功，包含 {len(df)} 个钙爆发事件"
        
        except Exception as e:
            return False, f"文件读取错误: {str(e)}"
    
    def run_analysis(self, file_path: str, output_dir: str, parameters: Dict[str, Any], 
                    additional_files: Dict[str, str] = None) -> Dict[str, Any]:
        self.prepare_output_dir(output_dir)
        
        # 构建命令
        script_path = os.path.join(ALGORITHM_DIR, 'cluster', 'cluster.py')
        cmd = ['python', script_path, '--input', file_path, '--output', output_dir]
        
        # 添加参数
        if parameters.get('k'):
            cmd.extend(['--k', str(parameters['k'])])
        
        if parameters.get('compare'):
            cmd.extend(['--compare', str(parameters['compare'])])
        
        try:
            # 运行分析
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
            
            if result.returncode == 0:
                # 扫描输出文件
                output_files = []
                for file in os.listdir(output_dir):
                    if file.endswith(('.png', '.jpg', '.xlsx', '.csv', '.txt')):
                        output_files.append({
                            'filename': file,
                            'path': os.path.join(output_dir, file),
                            'type': 'image' if file.endswith(('.png', '.jpg')) else 'data'
                        })
                
                return {
                    'success': True,
                    'output_files': output_files,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or result.stdout,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class HeatmapAdapter(ModuleAdapter):
    """热力图分析模块适配器"""
    
    def __init__(self):
        super().__init__('heatmap')
    
    def validate_input(self, file_path: str, additional_files: Dict[str, str] = None) -> tuple:
        try:
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            
            required_cols = ['stamp', 'behavior']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # 检查神经元列
            neuron_cols = [col for col in df.columns if 'neuron' in col.lower() or col.startswith('n') or 'Neuron_' in col]
            if not neuron_cols:
                missing_cols.append('神经元数据列')
            
            if missing_cols:
                return False, f"缺少必需的列: {', '.join(missing_cols)}"
            
            return True, f"验证成功，包含 {len(df)} 个时间点，{len(neuron_cols)} 个神经元"
        
        except Exception as e:
            return False, f"文件读取错误: {str(e)}"
    
    def run_analysis(self, file_path: str, output_dir: str, parameters: Dict[str, Any], 
                    additional_files: Dict[str, str] = None) -> Dict[str, Any]:
        self.prepare_output_dir(output_dir)
        
        # 构建命令
        script_path = os.path.join(ALGORITHM_DIR, 'heatmap', 'heatmap_behavior.py')
        cmd = ['python', script_path, '--input', file_path, '--output-dir', output_dir]
        
        # 添加参数
        if parameters.get('start_behavior'):
            cmd.extend(['--start-behavior', parameters['start_behavior']])
        
        if parameters.get('end_behavior'):
            cmd.extend(['--end-behavior', parameters['end_behavior']])
        
        if parameters.get('pre_behavior_time'):
            cmd.extend(['--pre-behavior-time', str(parameters['pre_behavior_time'])])
        
        if parameters.get('sorting_method'):
            cmd.extend(['--sorting-method', parameters['sorting_method']])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
            
            if result.returncode == 0:
                output_files = []
                for file in os.listdir(output_dir):
                    if file.endswith(('.png', '.jpg', '.csv')):
                        output_files.append({
                            'filename': file,
                            'path': os.path.join(output_dir, file),
                            'type': 'image' if file.endswith(('.png', '.jpg')) else 'data'
                        })
                
                return {
                    'success': True,
                    'output_files': output_files,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or result.stdout,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class TraceAdapter(ModuleAdapter):
    """轨迹图模块适配器"""
    
    def __init__(self):
        super().__init__('trace')
    
    def validate_input(self, file_path: str, additional_files: Dict[str, str] = None) -> tuple:
        try:
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            
            required_cols = ['stamp']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # 检查神经元列
            neuron_cols = [col for col in df.columns if 'neuron' in col.lower() or col.startswith('n') or 'Neuron_' in col]
            if not neuron_cols:
                missing_cols.append('神经元数据列')
            
            if missing_cols:
                return False, f"缺少必需的列: {', '.join(missing_cols)}"
            
            return True, f"验证成功，包含 {len(df)} 个时间点，{len(neuron_cols)} 个神经元"
        
        except Exception as e:
            return False, f"文件读取错误: {str(e)}"
    
    def run_analysis(self, file_path: str, output_dir: str, parameters: Dict[str, Any], 
                    additional_files: Dict[str, str] = None) -> Dict[str, Any]:
        self.prepare_output_dir(output_dir)
        
        # 构建命令
        script_path = os.path.join(ALGORITHM_DIR, 'trace', 'show_trace.py')
        cmd = ['python', script_path, '--input', file_path, '--output-dir', output_dir]
        
        # 添加参数
        if parameters.get('stamp_min') is not None:
            cmd.extend(['--stamp-min', str(parameters['stamp_min'])])
        
        if parameters.get('stamp_max') is not None:
            cmd.extend(['--stamp-max', str(parameters['stamp_max'])])
        
        if parameters.get('sort_method'):
            cmd.extend(['--sort-method', parameters['sort_method']])
        
        if parameters.get('max_neurons'):
            cmd.extend(['--max-neurons', str(parameters['max_neurons'])])
        
        if parameters.get('scaling_factor'):
            cmd.extend(['--scaling', str(parameters['scaling_factor'])])
        
        if parameters.get('line_width'):
            cmd.extend(['--line-width', str(parameters['line_width'])])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(script_path))
            
            if result.returncode == 0:
                output_files = []
                for file in os.listdir(output_dir):
                    if file.endswith(('.png', '.jpg')):
                        output_files.append({
                            'filename': file,
                            'path': os.path.join(output_dir, file),
                            'type': 'image'
                        })
                
                return {
                    'success': True,
                    'output_files': output_files,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
            else:
                return {
                    'success': False,
                    'error': result.stderr or result.stdout,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class EffectSizeAdapter(ModuleAdapter):
    """效应量计算模块适配器"""
    
    def __init__(self):
        super().__init__('effect_size')
    
    def validate_input(self, file_path: str, additional_files: Dict[str, str] = None) -> tuple:
        try:
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            
            required_cols = ['behavior']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # 检查神经元列
            neuron_cols = [col for col in df.columns if 'neuron' in col.lower() or col.startswith('n') or 'Neuron_' in col]
            if not neuron_cols:
                missing_cols.append('神经元数据列')
            
            if missing_cols:
                return False, f"缺少必需的列: {', '.join(missing_cols)}"
            
            return True, f"验证成功，包含 {len(df)} 个样本，{len(neuron_cols)} 个神经元"
        
        except Exception as e:
            return False, f"文件读取错误: {str(e)}"
    
    def run_analysis(self, file_path: str, output_dir: str, parameters: Dict[str, Any], 
                    additional_files: Dict[str, str] = None) -> Dict[str, Any]:
        self.prepare_output_dir(output_dir)
        
        try:
            # 直接调用效应量计算器
            sys.path.append(os.path.join(ALGORITHM_DIR, 'effect_size'))
            from effect_size_calculator import load_and_calculate_effect_sizes
            
            results = load_and_calculate_effect_sizes(
                neuron_data_path=file_path,
                behavior_col=parameters.get('behavior_column', 'behavior'),
                output_dir=output_dir
            )
            
            output_files = []
            for file_type, file_path in results['output_files'].items():
                if os.path.exists(file_path):
                    filename = os.path.basename(file_path)
                    output_files.append({
                        'filename': filename,
                        'path': file_path,
                        'type': 'image' if filename.endswith(('.png', '.jpg')) else 'data'
                    })
            
            return {
                'success': True,
                'output_files': output_files,
                'results': results
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class PrincipalNeuronAdapter(ModuleAdapter):
    """关键神经元分析模块适配器"""
    
    def __init__(self):
        super().__init__('principal_neuron')
    
    def validate_input(self, file_path: str, additional_files: Dict[str, str] = None) -> tuple:
        try:
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)
            
            required_cols = ['behavior']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            # 检查神经元列
            neuron_cols = [col for col in df.columns if 'neuron' in col.lower() or col.startswith('n') or 'Neuron_' in col]
            if not neuron_cols:
                missing_cols.append('神经元数据列')
            
            # 检查位置文件
            if not additional_files or 'position_file' not in additional_files:
                missing_cols.append('神经元位置文件')
            
            if missing_cols:
                return False, f"缺少必需的文件或列: {', '.join(missing_cols)}"
            
            return True, f"验证成功，包含 {len(df)} 个样本，{len(neuron_cols)} 个神经元"
        
        except Exception as e:
            return False, f"文件读取错误: {str(e)}"
    
    def run_analysis(self, file_path: str, output_dir: str, parameters: Dict[str, Any], 
                    additional_files: Dict[str, str] = None) -> Dict[str, Any]:
        self.prepare_output_dir(output_dir)
        
        try:
            # 准备数据文件
            import shutil
            
            # 创建临时数据目录
            temp_data_dir = os.path.join(output_dir, 'temp_data')
            os.makedirs(temp_data_dir, exist_ok=True)
            
            # 复制主数据文件
            main_data_file = os.path.join(temp_data_dir, 'main_data.xlsx')
            shutil.copy2(file_path, main_data_file)
            
            # 复制位置文件
            if additional_files and 'position_file' in additional_files:
                position_file = os.path.join(temp_data_dir, 'position_data.csv')
                shutil.copy2(additional_files['position_file'], position_file)
            
            # 修改配置并运行分析
            sys.path.append(os.path.join(ALGORITHM_DIR, 'principal_neuron', 'src'))
            
            # 这里需要根据实际的主分析脚本进行调用
            # 由于原始脚本比较复杂，我们简化处理
            script_path = os.path.join(ALGORITHM_DIR, 'principal_neuron', 'src', 'main_emtrace01_analysis.py')
            
            # 构建环境变量来传递参数
            env = os.environ.copy()
            env['ANALYSIS_OUTPUT_DIR'] = output_dir
            env['ANALYSIS_DATA_FILE'] = main_data_file
            env['ANALYSIS_POSITION_FILE'] = position_file
            
            if parameters.get('effect_size_threshold'):
                env['EFFECT_SIZE_THRESHOLD'] = str(parameters['effect_size_threshold'])
            
            result = subprocess.run(
                ['python', script_path],
                capture_output=True,
                text=True,
                cwd=os.path.dirname(script_path),
                env=env
            )
            
            # 扫描输出文件
            output_files = []
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file.endswith(('.png', '.jpg', '.csv', '.txt')) and not file.startswith('temp_'):
                        file_path = os.path.join(root, file)
                        output_files.append({
                            'filename': file,
                            'path': file_path,
                            'type': 'image' if file.endswith(('.png', '.jpg')) else 'data'
                        })
            
            return {
                'success': True,
                'output_files': output_files,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
        
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# 适配器工厂
class AdapterFactory:
    """适配器工厂类"""
    
    _adapters = {
        'cluster': ClusterAdapter,
        'heatmap': HeatmapAdapter,
        'trace': TraceAdapter,
        'effect_size': EffectSizeAdapter,
        'principal_neuron': PrincipalNeuronAdapter
    }
    
    @classmethod
    def get_adapter(cls, module_name: str) -> ModuleAdapter:
        """获取指定模块的适配器"""
        adapter_class = cls._adapters.get(module_name)
        if adapter_class:
            return adapter_class()
        else:
            raise ValueError(f"未知的模块: {module_name}")
    
    @classmethod
    def get_available_modules(cls) -> List[str]:
        """获取所有可用的模块名称"""
        return list(cls._adapters.keys())