import numpy as np
import pandas as pd
import os
from typing import Tuple, List, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder


class PathConfig:
    """
    路径配置类：集中管理所有输入输出路径配置
    
    在这里统一修改所有文件路径，便于管理和维护
    
    使用方法：
    --------
    1. 修改 default_data_file 来改变默认数据文件
    2. 修改 output_dir 路径来改变输出目录
    3. 在 alternative_data_files 中添加或修改其他数据文件
    4. 修改 default_behavior_column 来指定行为标签列名
    """
    
    def __init__(self):
        # 获取当前文件所在目录和项目根目录
        self.current_dir = os.path.dirname(os.path.abspath(__file__))
        self.principal_neuron_dir = os.path.dirname(self.current_dir)
        
        # === 输入数据路径配置 ===
        self.data_dir = os.path.join(self.principal_neuron_dir, "data")
        self.default_data_file = "bla6250EM0626goodtrace_plus.xlsx"  # 默认数据文件名
        self.default_data_path = os.path.join(self.data_dir, self.default_data_file)
        
        # === 输出路径配置 ===
        self.output_dir = os.path.join(self.principal_neuron_dir, "effect_size_output")
        self.effect_sizes_filename = "effect_sizes_bla6250EM0626goodtrace_plus.csv"  # 效应量结果文件名
        self.effect_sizes_path = os.path.join(self.output_dir, self.effect_sizes_filename)
        
        # # === 其他可选数据文件路径 ===
        # self.alternative_data_files = {
        #     "emtrace01": os.path.join(self.data_dir, "2980240924EMtrace.xlsx"),
        #     "emtrace02": os.path.join(self.data_dir, "62501monthgood0419fixedmerge.xlsx"),
        #     "emtrace03": os.path.join(self.data_dir, "bla6250EM0626goodtrace.xlsx")
        # }
        
        # === 行为标签配置 ===
        self.default_behavior_column = "behavior"  # None表示使用最后一列
        
    def get_data_path(self, data_key: str = "default") -> str:
        """
        获取数据文件路径
        
        参数
        ----------
        data_key : str
            数据文件键名，可选值：
            - "default": 使用默认数据文件
            - "emtrace01", "emtrace02", "emtrace03": 使用替代数据文件
            - 或直接传入完整文件路径
        
        返回
        ----------
        str
            数据文件的完整路径
        """
        if data_key == "default":
            return self.default_data_path
        elif data_key in self.alternative_data_files:
            return self.alternative_data_files[data_key]
        elif os.path.exists(data_key):  # 如果是完整路径且文件存在
            return data_key
        else:
            raise ValueError(f"无效的数据文件键名或路径: {data_key}")
    
    def ensure_output_dir(self) -> str:
        """
        确保输出目录存在，如果不存在则创建
        
        返回
        ----------
        str
            输出目录路径
        """
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir
    
    def get_output_path(self, filename: str = None) -> str:
        """
        获取输出文件的完整路径
        
        参数
        ----------
        filename : str, 可选
            输出文件名，如果未提供则使用默认的效应量文件名
        
        返回
        ----------
        str
            输出文件的完整路径
        """
        self.ensure_output_dir()
        if filename is None:
            filename = self.effect_sizes_filename
        return os.path.join(self.output_dir, filename)


# 创建全局路径配置实例
PATH_CONFIG = PathConfig()


class EffectSizeCalculator:
    """
    效应量计算器类：用于计算神经元活动与行为之间的Cohen's d效应量
    
    该类实现了从原始神经元活动数据计算效应量的完整流程，
    包括数据预处理、标准化、以及Cohen's d效应量计算，自动删除包含NaN值的行
    
    参数
    ----------
    behavior_labels : List[str], 可选
        行为标签列表，如果未提供将从数据中自动推断
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
        检测并删除包含NaN值的行
        
        参数
        ----------
        neuron_data : np.ndarray
            神经元活动数据
        behavior_data : np.ndarray
            行为标签数据
            
        返回
        ----------
        cleaned_neuron_data : np.ndarray
            删除NaN行后的神经元数据
        cleaned_behavior_data : np.ndarray
            删除NaN行后的行为数据
        """
        print("\n检测并删除包含NaN值的行...")
        
        # 检测神经元数据中的NaN值
        try:
            neuron_nan_mask = np.isnan(neuron_data).any(axis=1)
        except TypeError:
            # 如果神经元数据包含非数值类型，转换为浮点数
            print("警告: 神经元数据包含非数值类型，尝试转换为数值类型...")
            try:
                neuron_data = pd.DataFrame(neuron_data).apply(pd.to_numeric, errors='coerce').values
                neuron_nan_mask = np.isnan(neuron_data).any(axis=1)
            except Exception as e:
                print(f"神经元数据转换失败: {e}")
                neuron_nan_mask = np.zeros(neuron_data.shape[0], dtype=bool)
        
        # 检测行为数据中的NaN值
        try:
            if isinstance(behavior_data, pd.Series):
                behavior_nan_mask = pd.isna(behavior_data)
            else:
                # 对于numpy数组，需要区分数值类型和字符串类型
                if behavior_data.dtype.kind in ['U', 'S', 'O']:  # 字符串或对象类型
                    behavior_nan_mask = pd.isna(pd.Series(behavior_data))
                else:
                    # 数值类型
                    behavior_nan_mask = np.isnan(behavior_data)
        except Exception as e:
            print(f"检测行为数据NaN值时出错: {e}")
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
        
        # 统计NaN值信息
        self.nan_info['original_shape'] = neuron_data.shape
        self.nan_info['neuron_nan_rows'] = np.sum(neuron_nan_mask)
        self.nan_info['behavior_nan_rows'] = np.sum(behavior_nan_mask)
        self.nan_info['total_nan_rows'] = np.sum(neuron_nan_mask | behavior_nan_mask)
        
        print(f"原始数据形状: {neuron_data.shape}")
        print(f"神经元数据中包含NaN的行数: {self.nan_info['neuron_nan_rows']}")
        print(f"行为数据中包含NaN的行数: {self.nan_info['behavior_nan_rows']}")
        print(f"总共包含NaN的行数: {self.nan_info['total_nan_rows']}")
        
        if self.nan_info['total_nan_rows'] == 0:
            print("数据中没有发现NaN值")
            return neuron_data, behavior_data
        
        # 直接删除包含NaN的行
        print("删除包含NaN值的行...")
        valid_mask = ~(neuron_nan_mask | behavior_nan_mask)
        cleaned_neuron_data = neuron_data[valid_mask]
        cleaned_behavior_data = behavior_data[valid_mask]
        
        # 更新处理后的信息
        self.nan_info['cleaned_shape'] = cleaned_neuron_data.shape
        self.nan_info['removed_rows'] = self.nan_info['original_shape'][0] - cleaned_neuron_data.shape[0]
        
        print(f"处理后数据形状: {cleaned_neuron_data.shape}")
        print(f"删除的行数: {self.nan_info['removed_rows']}")
        print("NaN值处理完成，数据已清理")
            
        return cleaned_neuron_data, cleaned_behavior_data

    def preprocess_data(self, neuron_data: np.ndarray, behavior_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理神经元活动数据和行为标签（自动删除NaN值）
        
        对输入的神经元活动数据进行NaN值删除和标准化处理，对行为标签进行编码，
        确保数据格式符合后续分析要求
        
        参数
        ----------
        neuron_data : np.ndarray
            神经元活动数据，形状为 (n_samples, n_neurons)
        behavior_data : np.ndarray
            行为标签数据，形状为 (n_samples,)
            
        返回
        ----------
        X_scaled : np.ndarray
            标准化后的神经元活动数据
        y_encoded : np.ndarray
            编码后的行为标签
        """
        print("\n预处理神经元活动数据和行为标签...")
        
        # 删除包含NaN值的行
        cleaned_neuron_data, cleaned_behavior_data = self.remove_nan_rows(neuron_data, behavior_data)
        
        # 检查数据是否为空
        if cleaned_neuron_data.shape[0] == 0:
            raise ValueError("删除NaN值后没有剩余数据，请检查数据质量")
        
        # 标准化神经元活动数据
        X_scaled = self.scaler.fit_transform(cleaned_neuron_data)
        print(f"神经元数据标准化完成: {X_scaled.shape}")
        
        # 检查标准化后是否产生新的NaN值
        if np.isnan(X_scaled).any():
            print("警告: 标准化过程中产生了NaN值，可能是由于某些特征的标准差为0")
            # 用0替换标准化后的NaN值
            X_scaled = np.where(np.isnan(X_scaled), 0, X_scaled)
            print("已将标准化后的NaN值替换为0")
        
        # 编码行为标签
        y_encoded = self.label_encoder.fit_transform(cleaned_behavior_data)
        
        # 保存行为标签
        if self.behavior_labels is None:
            self.behavior_labels = self.label_encoder.classes_
        
        print(f"行为标签编码完成: {len(self.behavior_labels)} 个类别")
        for i, label in enumerate(self.behavior_labels):
            count = np.sum(y_encoded == i)
            print(f"  {label}: {count} 个样本")
            
        return X_scaled, y_encoded
    
    def calculate_effect_sizes(self, X_scaled: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算每个神经元对每种行为的Cohen's d效应量（增强NaN值处理）
        
        对于每种行为，计算该行为下神经元活动与其他行为下神经元活动的差异，
        使用Cohen's d公式量化这种差异的大小，并处理计算过程中可能出现的NaN值
        
        参数
        ----------
        X_scaled : np.ndarray
            标准化后的神经元活动数据
        y : np.ndarray
            编码后的行为标签
            
        返回
        ----------
        effect_sizes : Dict[str, np.ndarray]
            每种行为的效应量数组，键为行为名称，值为该行为下各神经元的效应量
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
            
            # 计算均值和标准差，使用nanmean和nanstd以防万一
            behavior_mean = np.nanmean(behavior_data, axis=0)
            other_mean = np.nanmean(other_data, axis=0)
            behavior_std = np.nanstd(behavior_data, axis=0, ddof=1)
            other_std = np.nanstd(other_data, axis=0, ddof=1)
            
            # 检查是否有NaN值
            if np.isnan(behavior_mean).any() or np.isnan(other_mean).any():
                print(f"警告: 行为 '{behavior}' 的均值计算中包含NaN值")
                
            if np.isnan(behavior_std).any() or np.isnan(other_std).any():
                print(f"警告: 行为 '{behavior}' 的标准差计算中包含NaN值")
            
            # 计算合并标准差 (Pooled Standard Deviation)
            pooled_std = np.sqrt((behavior_std**2 + other_std**2) / 2)
            
            # 处理NaN和零值情况
            # 如果pooled_std为NaN，用较大的标准差替代
            nan_mask = np.isnan(pooled_std)
            if nan_mask.any():
                print(f"警告: 行为 '{behavior}' 的合并标准差中有 {np.sum(nan_mask)} 个NaN值")
                pooled_std_fixed = np.where(nan_mask, np.nanmax([behavior_std, other_std], axis=0), pooled_std)
                pooled_std = pooled_std_fixed
            
            # 避免除零错误，同时处理NaN
            pooled_std = np.where((pooled_std == 0) | np.isnan(pooled_std), 1e-10, pooled_std)
            
            # 计算Cohen's d效应量
            effect_size = np.abs(behavior_mean - other_mean) / pooled_std
            
            # 处理效应量中的NaN值
            nan_effect_mask = np.isnan(effect_size)
            if nan_effect_mask.any():
                print(f"警告: 行为 '{behavior}' 的效应量中有 {np.sum(nan_effect_mask)} 个NaN值，将其设为0")
                effect_size = np.where(nan_effect_mask, 0, effect_size)
            
            effect_sizes[behavior] = effect_size
            
            print(f"  行为 '{behavior}': 平均效应量 = {np.mean(effect_size):.4f}")
            print(f"  最大效应量 = {np.max(effect_size):.4f}, 最小效应量 = {np.min(effect_size):.4f}")
        
        return effect_sizes
    
    def identify_key_neurons(self, effect_sizes: Dict[str, np.ndarray], threshold: float = 0.4) -> Dict[str, List[int]]:
        """
        基于效应量阈值识别关键神经元
        
        对于每种行为，找出效应量超过阈值的神经元，
        这些神经元被认为是该行为的关键判别神经元
        
        参数
        ----------
        effect_sizes : Dict[str, np.ndarray]
            每种行为的效应量数组
        threshold : float, 默认 0.4
            效应量阈值，超过此阈值的神经元被认为是关键神经元
            
        返回
        ----------
        key_neurons : Dict[str, List[int]]
            每种行为的关键神经元ID列表（1-based索引）
        """
        print(f"\n基于阈值 {threshold} 识别关键神经元...")
        
        key_neurons = {}
        
        for behavior, effect_size_array in effect_sizes.items():
            # 找出超过阈值的神经元
            significant_indices = np.where(effect_size_array >= threshold)[0]
            # 转换为1-based索引
            key_neuron_ids = [idx + 1 for idx in significant_indices]
            key_neurons[behavior] = sorted(key_neuron_ids)
            
            print(f"行为 '{behavior}': {len(key_neuron_ids)} 个关键神经元")
            if len(key_neuron_ids) > 0:
                print(f"  神经元ID: {key_neuron_ids}")
                # 显示对应的效应量
                effect_values = effect_size_array[significant_indices]
                print(f"  效应量: {[f'{val:.4f}' for val in effect_values]}")
            else:
                print(f"  没有神经元的效应量超过阈值 {threshold}")
        
        return key_neurons
    
    def export_effect_sizes_to_csv(self, effect_sizes: Dict[str, np.ndarray], 
                                  output_path: str, neuron_ids: List[int] = None) -> str:
        """
        将效应量结果导出为CSV文件
        
        将计算得到的效应量数据整理成表格格式并保存为CSV文件，
        便于后续分析和可视化使用
        
        参数
        ----------
        effect_sizes : Dict[str, np.ndarray]
            每种行为的效应量数组
        output_path : str
            输出CSV文件路径
        neuron_ids : List[int], 可选
            神经元ID列表，如果未提供则使用连续编号
            
        返回
        ----------
        str
            实际保存的文件路径
        """
        print(f"\n导出效应量数据到: {output_path}")
        
        # 准备数据
        if neuron_ids is None:
            # 假设神经元数量等于第一个行为的效应量数组长度
            first_behavior = list(effect_sizes.keys())[0]
            n_neurons = len(effect_sizes[first_behavior])
            neuron_ids = list(range(1, n_neurons + 1))
        
        # 创建DataFrame
        data_dict = {}
        for behavior, effect_array in effect_sizes.items():
            for i, neuron_id in enumerate(neuron_ids):
                if i < len(effect_array):
                    data_dict[f'Neuron_{neuron_id}'] = data_dict.get(f'Neuron_{neuron_id}', {})
                    data_dict[f'Neuron_{neuron_id}'][behavior] = effect_array[i]
        
        # 转换为DataFrame格式
        rows = []
        for behavior in effect_sizes.keys():
            row = {'Behavior': behavior}
            for neuron_key in sorted(data_dict.keys(), key=lambda x: int(x.split('_')[1])):
                row[neuron_key] = data_dict[neuron_key].get(behavior, 0.0)
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # 保存CSV文件
        df.to_csv(output_path, index=False)
        print(f"效应量数据已保存: {df.shape[0]} 行 x {df.shape[1]} 列")
        
        return output_path
    
    def get_top_neurons_per_behavior(self, effect_sizes: Dict[str, np.ndarray], 
                                   top_n: int = 10) -> Dict[str, Dict[str, Any]]:
        """
        获取每种行为的top-N关键神经元
        
        对于每种行为，按效应量从大到小排序，
        返回效应量最高的前N个神经元及其效应量值
        
        参数
        ----------
        effect_sizes : Dict[str, np.ndarray]
            每种行为的效应量数组
        top_n : int, 默认 10
            每种行为返回的top神经元数量
            
        返回
        ----------
        top_neurons : Dict[str, Dict[str, Any]]
            每种行为的top神经元信息，包含神经元ID和效应量
        """
        print(f"\n获取每种行为的top-{top_n}关键神经元...")
        
        top_neurons = {}
        
        for behavior, effect_array in effect_sizes.items():
            # 按效应量从大到小排序
            sorted_indices = np.argsort(effect_array)[::-1]
            top_indices = sorted_indices[:top_n]
            
            # 转换为1-based索引
            top_neuron_ids = [idx + 1 for idx in top_indices]
            top_effect_values = effect_array[top_indices]
            
            top_neurons[behavior] = {
                'neuron_ids': top_neuron_ids,
                'effect_sizes': top_effect_values.tolist(),
                'mean_effect_size': float(np.mean(top_effect_values))
            }
            
            print(f"行为 '{behavior}' top-{top_n} 神经元:")
            for i, (neuron_id, effect_val) in enumerate(zip(top_neuron_ids, top_effect_values)):
                print(f"  {i+1:2d}. 神经元 {neuron_id:2d}: 效应量 = {effect_val:.4f}")
        
        return top_neurons
    
    def calculate_effect_sizes_from_raw_data(self, neuron_data: np.ndarray, 
                                           behavior_data: np.ndarray) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        从原始数据计算效应量的完整流程
        
        这是一个便捷方法，整合了数据预处理和效应量计算的完整流程，
        适用于直接从原始数据开始的分析场景
        
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


def load_and_calculate_effect_sizes(neuron_data_path: str = None, behavior_col: str = None, 
                                  output_dir: str = None) -> Dict[str, Any]:
    """
    从文件加载数据并计算效应量的便捷函数（自动删除NaN值）
    
    这是一个高级接口函数，直接从Excel或CSV文件加载数据，
    自动执行完整的效应量计算流程，自动删除包含NaN值的行
    
    参数
    ----------
    neuron_data_path : str, 可选
        包含神经元数据的文件路径（Excel或CSV格式）
        如果未指定，将使用PATH_CONFIG中的默认数据路径
    behavior_col : str, 可选
        行为标签列名，如果未指定则使用最后一列
    output_dir : str, 可选
        输出目录路径，如果未指定则使用PATH_CONFIG中的默认输出目录
        
    返回
    ----------
    results : Dict[str, Any]
        包含效应量计算结果的字典
    """
    # 使用路径配置获取数据文件路径
    if neuron_data_path is None:
        neuron_data_path = PATH_CONFIG.get_data_path("default")
    
    print(f"从文件加载数据: {neuron_data_path}")
    print("NaN值处理策略: 自动删除包含NaN值的行")
    
    # 使用路径配置获取输出目录
    if output_dir is None:
        output_dir = PATH_CONFIG.ensure_output_dir()
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    if neuron_data_path.endswith('.xlsx'):
        data = pd.read_excel(neuron_data_path)
    elif neuron_data_path.endswith('.csv'):
        data = pd.read_csv(neuron_data_path)
    else:
        raise ValueError("不支持的文件格式，请使用Excel (.xlsx)或CSV (.csv)文件")
    
    print(f"数据加载完成: {data.shape}")
    
    # 检查加载的数据中是否有NaN值
    total_nan = data.isnull().sum().sum()
    if total_nan > 0:
        print(f"数据中发现 {total_nan} 个NaN值")
        nan_cols = data.columns[data.isnull().any()].tolist()
        print(f"包含NaN值的列: {nan_cols}")
    else:
        print("数据中没有发现NaN值")
    
    # 分离神经元数据和行为标签
    if behavior_col is None:
        # 使用最后一列作为行为标签
        neuron_data = data.iloc[:, :-1].values
        behavior_data = data.iloc[:, -1].values
        print(f"使用最后一列 '{data.columns[-1]}' 作为行为标签")
    else:
        if behavior_col not in data.columns:
            raise ValueError(f"指定的行为标签列 '{behavior_col}' 不存在")
        neuron_data = data.drop(columns=[behavior_col]).values
        behavior_data = data[behavior_col].values
        print(f"使用列 '{behavior_col}' 作为行为标签")
    
    print(f"神经元数据: {neuron_data.shape}, 行为数据: {behavior_data.shape}")
    
    # 创建效应量计算器并计算
    calculator = EffectSizeCalculator()
    effect_sizes, X_scaled, y_encoded = calculator.calculate_effect_sizes_from_raw_data(
        neuron_data, behavior_data
    )
    
    # 导出效应量数据
    csv_path = PATH_CONFIG.get_output_path(PATH_CONFIG.effect_sizes_filename)
    calculator.export_effect_sizes_to_csv(effect_sizes, csv_path)
    
    # 获取top神经元
    top_neurons = calculator.get_top_neurons_per_behavior(effect_sizes, top_n=10)
    
    # 整理结果
    results = {
        'effect_sizes': effect_sizes,
        'behavior_labels': calculator.behavior_labels.tolist(),
        'top_neurons': top_neurons,
        'nan_info': calculator.nan_info,  # 添加NaN处理信息
        'processed_data': {
            'X_scaled': X_scaled,
            'y_encoded': y_encoded
        },
        'calculator': calculator,
        'output_files': {
            'effect_sizes_csv': csv_path
        }
    }
    
    print(f"\n效应量计算完成！结果已保存到 {output_dir}")
    
    # 输出NaN处理摘要
    if calculator.nan_info and calculator.nan_info.get('total_nan_rows', 0) > 0:
        print(f"\n=== NaN值处理摘要 ===")
        print(f"原始数据行数: {calculator.nan_info['original_shape'][0]}")
        print(f"包含NaN的行数: {calculator.nan_info['total_nan_rows']}")
        print(f"处理后数据行数: {calculator.nan_info['cleaned_shape'][0]}")
        print(f"删除的行数: {calculator.nan_info['removed_rows']}")
        print(f"数据保留率: {calculator.nan_info['cleaned_shape'][0]/calculator.nan_info['original_shape'][0]*100:.2f}%")
    
    return results


if __name__ == "__main__":
    """
    使用真实数据计算效应量的示例（自动删除NaN值）
    """
    print("效应量计算器 - 真实数据分析（自动删除NaN值）...")
    
    # 使用路径配置获取数据路径和输出目录
    real_data_path = PATH_CONFIG.get_data_path("default")
    output_dir = PATH_CONFIG.ensure_output_dir()
    
    print(f"数据文件路径: {real_data_path}")
    print(f"输出目录: {output_dir}")
    
    # 检查数据文件是否存在
    if not os.path.exists(real_data_path):
        print(f"错误: 数据文件不存在 - {real_data_path}")
        print(f"请确保 {PATH_CONFIG.default_data_file} 文件位于 {PATH_CONFIG.data_dir} 目录下")
        print("或者修改 PATH_CONFIG 中的路径配置")
        exit(1)
    
    try:
        # 使用便捷函数加载和计算效应量
        results = load_and_calculate_effect_sizes(
            neuron_data_path=real_data_path,
            behavior_col=PATH_CONFIG.default_behavior_column,  # 使用配置的行为标签列
            output_dir=output_dir
        )
        
        print("\n=== 效应量计算结果汇总 ===")
        print(f"行为类别: {results['behavior_labels']}")
        
        # 显示每种行为的关键神经元信息
        for behavior, info in results['top_neurons'].items():
            print(f"\n行为 '{behavior}' 的前10个关键神经元:")
            print(f"  平均效应量: {info['mean_effect_size']:.4f}")
            print(f"  神经元ID: {info['neuron_ids']}")
            print(f"  对应效应量: {[f'{val:.4f}' for val in info['effect_sizes']]}")
        
        # 使用不同阈值识别关键神经元
        thresholds = [0.3, 0.4, 0.5, 0.6]
        print(f"\n=== 不同阈值下的关键神经元数量 ===")
        for threshold in thresholds:
            key_neurons = results['calculator'].identify_key_neurons(
                results['effect_sizes'], 
                threshold=threshold
            )
            print(f"阈值 {threshold}:")
            for behavior, neuron_ids in key_neurons.items():
                print(f"  {behavior}: {len(neuron_ids)} 个关键神经元")
        
        print(f"\n分析完成！详细结果已保存到:")
        for file_type, file_path in results['output_files'].items():
            print(f"  {file_type}: {file_path}")
            
    except Exception as e:
        print(f"处理过程中发生错误: {str(e)}")
        print("请检查数据文件格式是否正确")
        import traceback
        traceback.print_exc() 