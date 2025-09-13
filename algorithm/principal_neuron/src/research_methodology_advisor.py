import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ResearchMethodologyAdvisor:
    """
    基于神经元活动模式的研究方法建议系统
    """
    
    def __init__(self, temporal_patterns, clustering_result, response_patterns):
        """
        初始化研究方法建议器
        
        Args:
            temporal_patterns: 时间模式分析结果
            clustering_result: 聚类分析结果
            response_patterns: 响应模式识别结果
        """
        self.temporal_patterns = temporal_patterns
        self.clustering_result = clustering_result
        self.response_patterns = response_patterns
        
    def analyze_data_characteristics(self, df):
        """分析数据特征以指导研究方法选择"""
        print("=== 数据特征分析 ===")
        
        characteristics = {}
        
        # 基本数据特征
        n_timepoints = len(df)
        n_neurons = len([col for col in df.columns if col.startswith('Neuron_')])
        n_behaviors = len(df['Behavior'].unique())
        
        print(f"数据维度: {n_timepoints} 时间点 × {n_neurons} 神经元")
        print(f"行为类型数量: {n_behaviors}")
        
        # 数据完整性
        missing_data = df.isnull().sum().sum()
        print(f"缺失数据点: {missing_data}")
        
        # 时间序列特征
        behavior_sequence_length = n_timepoints
        behavior_transitions = n_timepoints - 1
        
        print(f"序列长度: {behavior_sequence_length}")
        print(f"行为转换次数: {behavior_transitions}")
        
        # 神经元活动特征
        neuron_cols = [col for col in df.columns if col.startswith('Neuron_')]
        activity_data = df[neuron_cols]
        
        mean_activity = activity_data.mean().mean()
        std_activity = activity_data.std().mean()
        activity_range = activity_data.max().max() - activity_data.min().min()
        
        print(f"平均神经元活动: {mean_activity:.3f}")
        print(f"活动标准差: {std_activity:.3f}")
        print(f"活动范围: {activity_range:.3f}")
        
        characteristics = {
            'n_timepoints': n_timepoints,
            'n_neurons': n_neurons,
            'n_behaviors': n_behaviors,
            'missing_data': missing_data,
            'sequence_length': behavior_sequence_length,
            'transitions': behavior_transitions,
            'mean_activity': mean_activity,
            'std_activity': std_activity,
            'activity_range': activity_range
        }
        
        return characteristics
    
    def suggest_research_paradigm(self, characteristics):
        """基于数据特征建议研究范式"""
        print("\n=== 研究范式建议 ===")
        
        paradigms = []
        
        # 1. 事件相关分析范式
        if characteristics['transitions'] > 5:
            paradigms.append({
                'name': '事件相关分析 (Event-Related Analysis)',
                'description': '分析特定行为事件前后的神经元活动变化',
                'approach': [
                    '定义关键行为事件（如行为转换时刻）',
                    '提取事件前后固定时间窗内的神经元活动',
                    '计算事件相关电位(ERP)或事件相关去同步化(ERD)',
                    '统计分析不同事件类型的神经反应差异'
                ],
                'advantages': '能够精确定位行为相关的神经活动',
                'suitable_for': '研究特定行为的神经机制'
            })
        
        # 2. 状态空间分析范式
        if characteristics['n_behaviors'] >= 3:
            paradigms.append({
                'name': '状态空间分析 (State Space Analysis)',
                'description': '将行为和神经活动建模为动态系统的状态转换',
                'approach': [
                    '定义行为状态和神经状态',
                    '建立状态转换矩阵',
                    '使用隐马尔可夫模型或卡尔曼滤波',
                    '分析状态转换的概率和神经基础'
                ],
                'advantages': '能够捕捉系统的动态特性和预测性',
                'suitable_for': '研究行为决策和状态切换机制'
            })
        
        # 3. 网络连接分析范式
        if characteristics['n_neurons'] >= 20:
            paradigms.append({
                'name': '功能网络分析 (Functional Network Analysis)',
                'description': '分析神经元间的功能连接和网络特性',
                'approach': [
                    '计算神经元间的功能连接（相关性、相干性等）',
                    '构建功能网络图',
                    '分析网络拓扑特征（聚类系数、路径长度等）',
                    '研究不同行为状态下的网络重组'
                ],
                'advantages': '揭示神经网络的组织原理和信息流',
                'suitable_for': '研究神经网络的功能架构'
            })
        
        # 4. 机器学习预测范式
        if characteristics['sequence_length'] > 10:
            paradigms.append({
                'name': '预测性建模 (Predictive Modeling)',
                'description': '使用机器学习模型预测行为或神经状态',
                'approach': [
                    '特征工程：提取时间、频率、统计特征',
                    '模型选择：支持向量机、随机森林、神经网络',
                    '时间序列建模：LSTM、GRU等循环神经网络',
                    '交叉验证和性能评估'
                ],
                'advantages': '量化神经活动对行为的预测能力',
                'suitable_for': '研究神经编码和解码机制'
            })
        
        # 打印建议的研究范式
        for i, paradigm in enumerate(paradigms, 1):
            print(f"\n{i}. {paradigm['name']}")
            print(f"   描述: {paradigm['description']}")
            print("   实施方法:")
            for step in paradigm['approach']:
                print(f"     • {step}")
            print(f"   优势: {paradigm['advantages']}")
            print(f"   适用于: {paradigm['suitable_for']}")
        
        return paradigms
    
    def suggest_specific_models(self, characteristics):
        """推荐具体的分析模型和参数"""
        print("\n=== 具体模型建议 ===")
        
        models = []
        
        # 1. 时间序列模型
        models.append({
            'category': '时间序列分析模型',
            'models': [
                {
                    'name': 'ARIMA模型',
                    'parameters': {
                        'p': '1-3 (自回归阶数)',
                        'd': '0-1 (差分次数)',
                        'q': '1-3 (移动平均阶数)'
                    },
                    'code_example': '''
from statsmodels.tsa.arima.model import ARIMA
# 对每个神经元建立ARIMA模型
model = ARIMA(neuron_activity, order=(2,1,2))
fitted_model = model.fit()
forecast = fitted_model.forecast(steps=5)
                    ''',
                    'interpretation': '分析神经元活动的时间依赖性和周期性'
                },
                {
                    'name': '隐马尔可夫模型(HMM)',
                    'parameters': {
                        'n_states': '3-7 (隐藏状态数)',
                        'covariance_type': 'full, diag, spherical',
                        'n_iter': '100-1000 (迭代次数)'
                    },
                    'code_example': '''
from hmmlearn import hmm
# 多元高斯HMM
model = hmm.GaussianHMM(n_components=5, covariance_type="full")
model.fit(neuron_data.values)
hidden_states = model.predict(neuron_data.values)
                    ''',
                    'interpretation': '识别潜在的神经活动状态和状态转换规律'
                }
            ]
        })
        
        # 2. 聚类分析模型
        models.append({
            'category': '聚类与降维模型',
            'models': [
                {
                    'name': '谱聚类',
                    'parameters': {
                        'n_clusters': f'{max(3, characteristics["n_neurons"]//10)}-{characteristics["n_neurons"]//5}',
                        'affinity': 'rbf, nearest_neighbors',
                        'gamma': '0.1-10.0'
                    },
                    'code_example': '''
from sklearn.cluster import SpectralClustering
clustering = SpectralClustering(n_clusters=5, affinity='rbf', gamma=1.0)
cluster_labels = clustering.fit_predict(correlation_matrix)
                    ''',
                    'interpretation': '基于相似性发现神经元功能模块'
                },
                {
                    'name': 'UMAP降维',
                    'parameters': {
                        'n_neighbors': '5-50',
                        'min_dist': '0.1-0.5',
                        'n_components': '2-10'
                    },
                    'code_example': '''
import umap
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
embedding = reducer.fit_transform(neuron_activity.T)
                    ''',
                    'interpretation': '在低维空间可视化神经元活动模式'
                }
            ]
        })
        
        # 3. 因果分析模型
        models.append({
            'category': '因果关系分析',
            'models': [
                {
                    'name': 'Granger因果检验',
                    'parameters': {
                        'maxlag': '1-5 (最大滞后阶数)',
                        'test': 'ssr_ftest, ssr_chi2test',
                        'verbose': 'True/False'
                    },
                    'code_example': '''
from statsmodels.tsa.stattools import grangercausalitytests
# 检验神经元A是否Granger因果神经元B
result = grangercausalitytests(data[['neuron_B', 'neuron_A']], maxlag=3)
                    ''',
                    'interpretation': '确定神经元间的因果关系方向'
                },
                {
                    'name': '动态因果模型(DCM)',
                    'parameters': {
                        'TR': '时间分辨率',
                        'model_space': 'full, sparse',
                        'estimation': 'VL, ML'
                    },
                    'code_example': '''
# 使用SPM或Python的DCM实现
# 定义连接矩阵和输入矩阵
A = np.zeros((n_regions, n_regions))  # 内在连接
B = np.zeros((n_regions, n_regions, n_inputs))  # 调制连接
C = np.zeros((n_regions, n_inputs))  # 输入连接
                    ''',
                    'interpretation': '建模神经网络的有效连接'
                }
            ]
        })
        
        # 4. 机器学习模型
        models.append({
            'category': '预测性机器学习',
            'models': [
                {
                    'name': 'LSTM神经网络',
                    'parameters': {
                        'hidden_size': '32-256',
                        'num_layers': '1-3',
                        'dropout': '0.1-0.5',
                        'sequence_length': '5-20'
                    },
                    'code_example': '''
import torch.nn as nn
class NeuralLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
                    ''',
                    'interpretation': '预测未来的行为状态或神经活动'
                }
            ]
        })
        
        # 打印模型建议
        for category_info in models:
            print(f"\n📊 {category_info['category']}")
            for model in category_info['models']:
                print(f"\n  🔧 {model['name']}")
                print("     参数建议:")
                for param, value in model['parameters'].items():
                    print(f"       {param}: {value}")
                print(f"     应用: {model['interpretation']}")
                print("     代码示例:")
                print(model['code_example'])
        
        return models
    
    def generate_analysis_workflow(self, characteristics):
        """生成完整的分析工作流程"""
        print("\n=== 建议的分析工作流程 ===")
        
        workflow = [
            {
                'step': 1,
                'name': '数据预处理与质量控制',
                'tasks': [
                    '检查数据完整性和异常值',
                    '标准化或归一化神经元活动数据',
                    '去除趋势和季节性成分（如适用）',
                    '数据平滑（移动平均或高斯滤波）'
                ],
                'tools': ['pandas', 'scipy.signal', 'sklearn.preprocessing'],
                'duration': '1-2天'
            },
            {
                'step': 2,
                'name': '探索性数据分析',
                'tasks': [
                    '可视化神经元活动时间序列',
                    '分析行为转换模式',
                    '计算神经元间相关性矩阵',
                    '识别活动峰值和异常模式'
                ],
                'tools': ['matplotlib', 'seaborn', 'plotly'],
                'duration': '2-3天'
            },
            {
                'step': 3,
                'name': '模式识别与聚类',
                'tasks': [
                    '执行层次聚类分析',
                    '进行主成分分析(PCA)',
                    '使用UMAP进行非线性降维',
                    '验证聚类结果的稳定性'
                ],
                'tools': ['scikit-learn', 'umap-learn', 'scipy.cluster'],
                'duration': '3-4天'
            },
            {
                'step': 4,
                'name': '时间动力学分析',
                'tasks': [
                    '拟合ARIMA时间序列模型',
                    '进行隐马尔可夫模型分析',
                    '计算神经元活动的自相关函数',
                    '分析周期性和趋势成分'
                ],
                'tools': ['statsmodels', 'hmmlearn', 'pykalman'],
                'duration': '4-5天'
            },
            {
                'step': 5,
                'name': '因果关系分析',
                'tasks': [
                    'Granger因果检验',
                    '构建有向图网络',
                    '分析信息流方向',
                    '验证因果关系的鲁棒性'
                ],
                'tools': ['statsmodels', 'networkx', 'causalgraphicalmodels'],
                'duration': '3-4天'
            },
            {
                'step': 6,
                'name': '预测建模',
                'tasks': [
                    '特征工程和选择',
                    '训练机器学习模型',
                    '交叉验证和超参数调优',
                    '模型解释和重要性分析'
                ],
                'tools': ['scikit-learn', 'tensorflow/pytorch', 'shap'],
                'duration': '5-7天'
            },
            {
                'step': 7,
                'name': '结果验证与解释',
                'tasks': [
                    '统计显著性检验',
                    '效应量计算',
                    '敏感性分析',
                    '生物学意义解释'
                ],
                'tools': ['scipy.stats', 'statsmodels', 'pingouin'],
                'duration': '3-4天'
            },
            {
                'step': 8,
                'name': '可视化与报告',
                'tasks': [
                    '创建综合分析图表',
                    '制作交互式可视化',
                    '撰写分析报告',
                    '准备学术论文图表'
                ],
                'tools': ['matplotlib', 'plotly', 'seaborn', 'bokeh'],
                'duration': '3-5天'
            }
        ]
        
        print("📋 完整分析工作流程:")
        total_duration = 0
        
        for step_info in workflow:
            print(f"\n步骤 {step_info['step']}: {step_info['name']} ({step_info['duration']})")
            print("    任务:")
            for task in step_info['tasks']:
                print(f"      • {task}")
            print(f"    推荐工具: {', '.join(step_info['tools'])}")
            
            # 提取持续时间的数字部分来计算总时间
            duration_parts = step_info['duration'].split('-')
            if len(duration_parts) == 2:
                avg_duration = (int(duration_parts[0]) + int(duration_parts[1].split('天')[0])) / 2
            else:
                avg_duration = int(duration_parts[0].split('天')[0])
            total_duration += avg_duration
        
        print(f"\n⏱️  预计总时间: {total_duration:.0f}天 ({total_duration/7:.1f}周)")
        
        return workflow
    
    def create_research_proposal_template(self):
        """生成研究提案模板"""
        print("\n=== 研究提案模板 ===")
        
        template = f"""
# 小鼠神经元活动时间模式研究提案

## 1. 研究背景与目标
**研究问题**: 小鼠在行为转换过程中神经元活动的时间动态模式
**具体目标**:
- 识别行为转换相关的神经活动模式
- 建立神经活动与行为状态的预测模型
- 揭示神经元网络的功能组织原理

## 2. 数据特征
- 时间点数量: {self.clustering_result.get('activity_matrix', pd.DataFrame()).shape[1] if self.clustering_result else 'N/A'}
- 神经元数量: {len(self.clustering_result.get('clusters', {}).get(1, [])) if self.clustering_result else 'N/A'}
- 行为类型: {len(self.response_patterns) if self.response_patterns else 'N/A'}

## 3. 方法学框架
### 3.1 时间序列分析
- **ARIMA模型**: 分析神经元活动的自回归特性
- **隐马尔可夫模型**: 识别潜在的神经状态
- **状态空间模型**: 跟踪动态变化过程

### 3.2 模式识别
- **聚类分析**: 识别功能相似的神经元群
- **主成分分析**: 降维和模式提取
- **谱聚类**: 基于相似性的网络划分

### 3.3 因果推断
- **Granger因果检验**: 确定神经元间的因果关系
- **动态因果模型**: 建模有效连接
- **信息论方法**: 量化信息传递

### 3.4 预测建模
- **机器学习**: 支持向量机、随机森林
- **深度学习**: LSTM、GRU时间序列模型
- **集成方法**: 提高预测准确性

## 4. 预期成果
1. **科学发现**:
   - 神经元活动的时间动力学规律
   - 行为转换的神经机制
   - 神经网络的功能模块

2. **方法学贡献**:
   - 多模态时间序列分析框架
   - 神经活动预测模型
   - 可重复的分析管道

3. **应用价值**:
   - 神经疾病的早期诊断
   - 脑机接口技术
   - 药物筛选平台

## 5. 时间安排
- **第1-2周**: 数据预处理与质量控制
- **第3-4周**: 探索性数据分析
- **第5-6周**: 模式识别与聚类
- **第7-8周**: 时间动力学分析
- **第9-10周**: 因果关系分析
- **第11-12周**: 预测建模
- **第13-14周**: 结果验证与解释
- **第15-16周**: 可视化与报告

## 6. 风险与对策
**潜在风险**:
- 数据质量问题 → 严格的质控流程
- 模型过拟合 → 交叉验证和正则化
- 计算资源限制 → 云计算平台

**质量保证**:
- 可重复性检验
- 多种方法验证
- 专家同行评议

## 7. 资源需求
**软件工具**: Python科学计算栈 (pandas, scikit-learn, tensorflow)
**硬件需求**: GPU加速计算 (深度学习模型)
**人力资源**: 数据科学家 + 神经科学专家
**时间投入**: 4个月全职工作量

## 8. 伦理考虑
- 动物实验伦理审查
- 数据使用授权
- 结果发布规范
        """
        
        print(template)
        
        # 保存到文件
        with open('research_proposal_template.md', 'w', encoding='utf-8') as f:
            f.write(template)
        
        print("\n📄 研究提案模板已保存到 'research_proposal_template.md'")
        
        return template

def create_methodology_report(data_path):
    """创建完整的方法学建议报告"""
    print("正在生成研究方法学建议报告...")
    
    # 加载数据进行分析
    from temporal_pattern_analysis import TemporalPatternAnalyzer
    
    analyzer = TemporalPatternAnalyzer(data_path)
    df = analyzer.load_and_preprocess_data()
    
    # 运行各种分析
    temporal_patterns = analyzer.analyze_sequential_patterns()
    clustering_result = analyzer.cluster_neuron_patterns()
    response_patterns = analyzer.identify_response_patterns()
    
    # 创建方法学建议器
    advisor = ResearchMethodologyAdvisor(temporal_patterns, clustering_result, response_patterns)
    
    # 分析数据特征
    characteristics = advisor.analyze_data_characteristics(df)
    
    # 生成各种建议
    paradigms = advisor.suggest_research_paradigm(characteristics)
    models = advisor.suggest_specific_models(characteristics)
    workflow = advisor.generate_analysis_workflow(characteristics)
    proposal = advisor.create_research_proposal_template()
    
    return {
        'characteristics': characteristics,
        'paradigms': paradigms,
        'models': models,
        'workflow': workflow,
        'proposal': proposal
    }

if __name__ == "__main__":
    data_path = 'data/EMtrace01-多标签版.csv'
    report = create_methodology_report(data_path) 