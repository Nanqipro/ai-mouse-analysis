#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
神经元数据分析Web工具测试脚本

测试各个模块的基本功能和API接口
"""

import os
import sys
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# 添加路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'adapters'))

from modules_config import MODULES_CONFIG, get_all_modules
from module_adapter import AdapterFactory

class WebToolTester:
    """Web工具测试类"""
    
    def __init__(self, base_url='http://localhost:5000'):
        self.base_url = base_url
        self.test_results = []
    
    def log_test(self, test_name, success, message):
        """记录测试结果"""
        result = {
            'test_name': test_name,
            'success': success,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}: {message}")
    
    def test_modules_config(self):
        """测试模块配置"""
        try:
            modules = get_all_modules()
            if len(modules) > 0:
                self.log_test("模块配置加载", True, f"成功加载 {len(modules)} 个模块")
            else:
                self.log_test("模块配置加载", False, "没有找到任何模块")
            
            # 测试每个模块的配置完整性
            for module_name in modules:
                config = MODULES_CONFIG[module_name]
                required_keys = ['name', 'description', 'script_path', 'parameters', 'outputs']
                missing_keys = [key for key in required_keys if key not in config]
                
                if not missing_keys:
                    self.log_test(f"模块配置-{module_name}", True, "配置完整")
                else:
                    self.log_test(f"模块配置-{module_name}", False, f"缺少配置项: {missing_keys}")
        
        except Exception as e:
            self.log_test("模块配置加载", False, str(e))
    
    def test_adapters(self):
        """测试模块适配器"""
        try:
            available_modules = AdapterFactory.get_available_modules()
            self.log_test("适配器工厂", True, f"支持 {len(available_modules)} 个模块")
            
            # 测试每个适配器的创建
            for module_name in available_modules:
                try:
                    adapter = AdapterFactory.get_adapter(module_name)
                    self.log_test(f"适配器-{module_name}", True, "创建成功")
                except Exception as e:
                    self.log_test(f"适配器-{module_name}", False, str(e))
        
        except Exception as e:
            self.log_test("适配器工厂", False, str(e))
    
    def create_test_data(self):
        """创建测试数据文件"""
        try:
            test_dir = os.path.join(os.path.dirname(__file__), 'test_data')
            os.makedirs(test_dir, exist_ok=True)
            
            # 创建简单的测试数据
            np.random.seed(42)
            
            # 神经元活动数据
            n_samples = 1000
            n_neurons = 20
            
            data = {
                'stamp': range(n_samples),
                'behavior': np.random.choice(['Close', 'Middle', 'Open'], n_samples)
            }
            
            # 添加神经元列
            for i in range(n_neurons):
                data[f'Neuron_{i+1}'] = np.random.normal(0, 1, n_samples)
            
            df = pd.DataFrame(data)
            test_file = os.path.join(test_dir, 'test_neuron_data.xlsx')
            df.to_excel(test_file, index=False)
            
            # 创建位置数据
            position_data = {
                'NeuronID': [f'Neuron_{i+1}' for i in range(n_neurons)],
                'X': np.random.uniform(0, 100, n_neurons),
                'Y': np.random.uniform(0, 100, n_neurons)
            }
            
            position_df = pd.DataFrame(position_data)
            position_file = os.path.join(test_dir, 'test_position_data.csv')
            position_df.to_csv(position_file, index=False)
            
            self.log_test("测试数据创建", True, f"创建了 {test_file} 和 {position_file}")
            return test_file, position_file
        
        except Exception as e:
            self.log_test("测试数据创建", False, str(e))
            return None, None
    
    def test_file_validation(self):
        """测试文件验证功能"""
        test_file, position_file = self.create_test_data()
        if not test_file:
            return
        
        try:
            # 测试每个模块的文件验证
            for module_name in get_all_modules():
                try:
                    adapter = AdapterFactory.get_adapter(module_name)
                    
                    if module_name == 'principal_neuron':
                        # 需要额外的位置文件
                        is_valid, message = adapter.validate_input(test_file, {'position_file': position_file})
                    else:
                        is_valid, message = adapter.validate_input(test_file)
                    
                    self.log_test(f"文件验证-{module_name}", is_valid, message)
                
                except Exception as e:
                    self.log_test(f"文件验证-{module_name}", False, str(e))
        
        except Exception as e:
            self.log_test("文件验证", False, str(e))
    
    def test_api_endpoints(self):
        """测试API端点"""
        endpoints = [
            ('/api/modules', 'GET'),
            ('/api/module/cluster', 'GET'),
        ]
        
        for endpoint, method in endpoints:
            try:
                url = f"{self.base_url}{endpoint}"
                if method == 'GET':
                    response = requests.get(url, timeout=5)
                else:
                    response = requests.post(url, timeout=5)
                
                if response.status_code == 200:
                    self.log_test(f"API-{endpoint}", True, f"状态码: {response.status_code}")
                else:
                    self.log_test(f"API-{endpoint}", False, f"状态码: {response.status_code}")
            
            except requests.exceptions.ConnectionError:
                self.log_test(f"API-{endpoint}", False, "无法连接到服务器")
            except Exception as e:
                self.log_test(f"API-{endpoint}", False, str(e))
    
    def test_directory_structure(self):
        """测试目录结构"""
        base_dir = os.path.dirname(__file__)
        required_dirs = [
            'config',
            'adapters',
            'templates',
            'static',
            'static/js',
            'static/css'
        ]
        
        required_files = [
            'app.py',
            'run.py',
            'requirements.txt',
            'README.md',
            'config/modules_config.py',
            'adapters/module_adapter.py',
            'templates/index.html',
            'static/js/app.js',
            'static/css/style.css'
        ]
        
        # 检查目录
        for dir_path in required_dirs:
            full_path = os.path.join(base_dir, dir_path)
            if os.path.exists(full_path) and os.path.isdir(full_path):
                self.log_test(f"目录-{dir_path}", True, "存在")
            else:
                self.log_test(f"目录-{dir_path}", False, "不存在")
        
        # 检查文件
        for file_path in required_files:
            full_path = os.path.join(base_dir, file_path)
            if os.path.exists(full_path) and os.path.isfile(full_path):
                self.log_test(f"文件-{file_path}", True, "存在")
            else:
                self.log_test(f"文件-{file_path}", False, "不存在")
    
    def run_all_tests(self):
        """运行所有测试"""
        print("="*60)
        print("神经元数据分析Web工具 - 功能测试")
        print("="*60)
        
        print("\n1. 测试目录结构...")
        self.test_directory_structure()
        
        print("\n2. 测试模块配置...")
        self.test_modules_config()
        
        print("\n3. 测试模块适配器...")
        self.test_adapters()
        
        print("\n4. 测试文件验证...")
        self.test_file_validation()
        
        print("\n5. 测试API端点...")
        self.test_api_endpoints()
        
        # 生成测试报告
        self.generate_report()
    
    def generate_report(self):
        """生成测试报告"""
        print("\n" + "="*60)
        print("测试报告")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = len([r for r in self.test_results if r['success']])
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests}")
        print(f"失败: {failed_tests}")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n失败的测试:")
            for result in self.test_results:
                if not result['success']:
                    print(f"  - {result['test_name']}: {result['message']}")
        
        # 保存详细报告
        report_file = os.path.join(os.path.dirname(__file__), 'test_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'success_rate': passed_tests/total_tests*100
                },
                'results': self.test_results
            }, f, ensure_ascii=False, indent=2)
        
        print(f"\n详细报告已保存到: {report_file}")

def main():
    """主函数"""
    tester = WebToolTester()
    tester.run_all_tests()

if __name__ == '__main__':
    main()