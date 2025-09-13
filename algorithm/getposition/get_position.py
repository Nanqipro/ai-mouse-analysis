"""
This script provides an interactive tool for marking points on an image and saving their relative coordinates.
The script allows users to:
1. Click left mouse button to add points
2. Click left mouse button multiple times quickly in same position to skip that many numbers
3. Click right mouse button to undo the last point
4. Drag existing points to adjust their positions
5. Close the window to save the points and view their relative positions
"""

import matplotlib
matplotlib.use('TkAgg')  # Use interactive backend

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from typing import List, Tuple, Dict, Optional
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.text import Text
import time
import os

# Configure matplotlib settings
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class PointMarker:
    def __init__(self, image_path: str, start_number: int = 1):
        """
        初始化 PointMarker，设置图像路径和相关参数
        
        Parameters
        ----------
        image_path : str
            图像文件路径
        start_number : int, optional
            起始编号，默认为1
        """
        self.image_path = image_path
        self.img = mpimg.imread(image_path)
        self.img_height, self.img_width = self.img.shape[0], self.img.shape[1]
        
        self.clicked_points: Dict[int, Tuple[float, float]] = {}  # 使用字典存储点的编号和坐标
        self.point_plots: Dict[int, Line2D] = {}  # 使用字典存储点的编号和图形对象
        self.text_labels: Dict[int, Text] = {}  # 使用字典存储点的编号和文本对象
        self.skipped_numbers: set = set()  # 存储被跳过的编号
        
        # 操作历史记录 - 记录所有操作以支持撤销
        self.operation_history: List[Dict] = []  # 操作历史列表
        
        self.current_number = start_number  # 当前编号，支持自定义起始值
        self.last_click_time = 0  # 用于检测双击
        self.last_click_pos = (None, None)  # 用于存储上次点击位置
        self.double_click_threshold = 0.3  # 双击时间阈值（秒），降低到0.3秒更容易触发
        
        # 拖动相关变量
        self.dragging = False
        self.drag_point_number: Optional[int] = None
        self.drag_threshold = 5  # 拖动判定阈值（像素）
        
        self.fig, self.ax = plt.subplots()
        self.setup_plot()
        
    def setup_plot(self) -> None:
        """设置初始绘图配置"""
        self.ax.imshow(self.img)
        self.ax.set_title(f'请用左键标点，右键撤销上一次操作（包括跳过），快速双击跳过编号，拖动可调整点位 (关闭窗口结束)\n当前编号: {self.current_number}')
        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
        
    def find_nearest_point(self, x: float, y: float) -> Optional[int]:
        """Find the number of the nearest point within threshold distance."""
        min_dist = float('inf')
        nearest_num = None
        
        for num, (px, py) in self.clicked_points.items():
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist and dist < self.drag_threshold:
                min_dist = dist
                nearest_num = num
                
        return nearest_num
        
    def onclick(self, event) -> None:
        """
        处理鼠标点击事件
        
        Parameters
        ----------
        event : MouseEvent
            包含点击信息的鼠标事件对象
        """
        if not event.inaxes:
            return
            
        if event.button == 1:
            current_time = time.time()
            current_pos = (event.xdata, event.ydata)
            
            # 检查是否是快速连续点击（双击）
            # 位置阈值调整为10像素，时间阈值为0.3秒
            is_consecutive_click = (current_time - self.last_click_time < self.double_click_threshold and
                             self.last_click_pos[0] is not None and
                             abs(current_pos[0] - self.last_click_pos[0]) < 10 and
                             abs(current_pos[1] - self.last_click_pos[1]) < 10)
            
            if is_consecutive_click:
                # 双击时跳过当前编号
                self._skip_current_number()
                self.last_click_time = current_time
                self.last_click_pos = current_pos
                return
                
            # 检查是否点击了已存在的点
            nearest_num = self.find_nearest_point(event.xdata, event.ydata)
            if nearest_num is not None:
                # 开始拖动已存在的点
                self.dragging = True
                self.drag_point_number = nearest_num
                return
                
            self._add_point(event.xdata, event.ydata)
            self.last_click_time = current_time
            self.last_click_pos = current_pos
            
        elif event.button == 3:
            self._undo_last_operation()
            
    def onrelease(self, event) -> None:
        """Handle mouse button release events."""
        if event.button == 1:
            self.dragging = False
            self.drag_point_number = None
            
    def onmotion(self, event) -> None:
        """Handle mouse motion events for dragging points."""
        if not self.dragging or self.drag_point_number is None or not event.inaxes:
            return
            
        # 更新点的位置
        self.clicked_points[self.drag_point_number] = (event.xdata, event.ydata)
        
        # 更新点的显示位置
        self.point_plots[self.drag_point_number].set_data([event.xdata], [event.ydata])
        
        # 更新标签位置
        self.text_labels[self.drag_point_number].set_position((event.xdata + 5, event.ydata))
        
        plt.draw()
        
    def _skip_current_number(self) -> None:
        """
        跳过当前编号并记录操作历史
        """
        skipped_number = self.current_number
        self.skipped_numbers.add(skipped_number)
        
        # 记录跳过操作到历史中
        operation = {
            'type': 'skip',
            'number': skipped_number,
            'previous_current_number': skipped_number
        }
        self.operation_history.append(operation)
        
        print(f"跳过编号 {skipped_number}")
        self._advance_to_next_available_number()
        # 更新标题显示当前编号
        self.ax.set_title(f'请用左键标点，右键撤销上一次操作（包括跳过），快速双击跳过编号，拖动可调整点位 (关闭窗口结束)\n当前编号: {self.current_number}')
        plt.draw()
        
    def _advance_to_next_available_number(self) -> None:
        """
        将当前编号推进到下一个可用的编号（跳过已被标记跳过的编号）
        """
        self.current_number += 1
        while self.current_number in self.skipped_numbers or self.current_number in self.clicked_points:
            self.current_number += 1
        
    def _add_point(self, x: float, y: float) -> None:
        """
        在图上添加新点并记录操作历史
        
        Parameters
        ----------
        x : float
            点的x坐标
        y : float
            点的y坐标
        """
        point_number = self.current_number
        self.clicked_points[point_number] = (x, y)
        point = self.ax.plot(x, y, 'ro', markersize=5)[0]
        self.point_plots[point_number] = point
        
        text = self.ax.text(x+5, y, str(point_number), color='white', 
                          fontsize=8, backgroundcolor='black')
        self.text_labels[point_number] = text
        
        # 记录添加点操作到历史中
        operation = {
            'type': 'add_point',
            'number': point_number,
            'coordinates': (x, y),
            'previous_current_number': point_number
        }
        self.operation_history.append(operation)
        
        print(f"已标记点 {point_number}")
        self._advance_to_next_available_number()
        # 更新标题显示当前编号
        self.ax.set_title(f'请用左键标点，右键撤销上一次操作（包括跳过），快速双击跳过编号，拖动可调整点位 (关闭窗口结束)\n当前编号: {self.current_number}')
        plt.draw()
        
    def _undo_last_operation(self) -> None:
        """撤销最后一个操作（可能是添加点或跳过编号）"""
        if not self.operation_history:
            print("没有可撤销的操作")
            return
            
        last_operation = self.operation_history.pop()
        
        if last_operation['type'] == 'add_point':
            # 撤销添加点操作
            number = last_operation['number']
            
            # 移除点和标签
            if number in self.point_plots:
                self.point_plots[number].remove()
                del self.point_plots[number]
                
            if number in self.text_labels:
                self.text_labels[number].remove()
                del self.text_labels[number]
                
            # 从字典中删除
            if number in self.clicked_points:
                del self.clicked_points[number]
            
            # 恢复当前编号
            self.current_number = number
            print(f"已撤销标记点 {number}")
            
        elif last_operation['type'] == 'skip':
            # 撤销跳过编号操作
            number = last_operation['number']
            
            # 从跳过集合中移除
            if number in self.skipped_numbers:
                self.skipped_numbers.remove(number)
                
            # 恢复当前编号
            self.current_number = number
            print(f"已撤销跳过编号 {number}")
        
        # 更新标题显示当前编号
        self.ax.set_title(f'请用左键标点，右键撤销上一次操作（包括跳过），快速双击跳过编号，拖动可调整点位 (关闭窗口结束)\n当前编号: {self.current_number}')
        plt.draw()
        
    def _remove_last_point(self) -> None:
        """
        移除最后添加的点（保留此方法以兼容性，实际调用_undo_last_operation）
        """
        self._undo_last_operation()
        
    def get_relative_coordinates(self) -> np.ndarray:
        """
        Convert clicked points to relative coordinates.
        
        Returns:
            np.ndarray: Array of relative coordinates with their numbers.
            Maintains original numbers (including gaps from skipped numbers).
        """
        if not self.clicked_points:
            return np.array([])
            
        # 将字典转换为有序列表，保持原始编号
        sorted_points = sorted(self.clicked_points.items())
        numbers = np.array([num for num, _ in sorted_points])
        coordinates = np.array([(x / self.img_width, y / self.img_height) 
                              for _, (x, y) in sorted_points])
        
        # 组合原始编号和相对坐标
        return np.column_stack((numbers, coordinates))
        
    def save_coordinates(self, output_file: str) -> None:
        """
        Save relative coordinates to a CSV file.
        Maintains original numbers (including gaps from skipped numbers).
        
        Args:
            output_file (str): Path to save the coordinates
        """
        relative_points = self.get_relative_coordinates()
        if len(relative_points) > 0:
            np.savetxt(output_file, relative_points, delimiter=',',
                      header='number,relative_x,relative_y', comments='')
            
            skipped_count = len(self.skipped_numbers)
            total_numbers_used = int(relative_points[-1,0]) if len(relative_points) > 0 else 0
            
            print(f"已保存标记点相对坐标至 {output_file}")
            print(f"共标记 {len(relative_points)} 个点，跳过 {skipped_count} 个编号")
            print(f"编号范围 {int(relative_points[0,0])} - {total_numbers_used}")
            if skipped_count > 0:
                print(f"跳过的编号: {sorted(self.skipped_numbers)}")
                
            self._plot_relative_coordinates(relative_points)
        else:
            print("未标记任何点，不生成输出文件及相对坐标系散点图。")
            
    def _plot_relative_coordinates(self, relative_points: np.ndarray) -> None:
        """
        Plot the relative coordinates in a scatter plot and save it.
        Shows the original numbers (maintaining gaps from skipped numbers).
        
        Args:
            relative_points (np.ndarray): Array of relative coordinates to plot
        """
        # 创建保存图像的目录
        graph_dir = os.path.join(os.path.dirname(os.path.dirname(self.image_path)), 'graph')
        os.makedirs(graph_dir, exist_ok=True)
        
        # 获取原始图像文件名（不含扩展名）
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        
        fig2, ax2 = plt.subplots()
        ax2.scatter(relative_points[:,1], relative_points[:,2], c='r', s=20)
        ax2.set_title('标记点在相对坐标系中的分布')
        ax2.set_xlabel('relative x')
        ax2.set_ylabel('relative y')
        ax2.set_xlim([0,1])
        ax2.set_ylim([1,0])
        ax2.grid(True)
        
        for num, rx, ry in relative_points:
            ax2.text(rx+0.01, ry, str(int(num)), color='red', fontsize=8)
            
        # 保存图像
        save_path = os.path.join(graph_dir, f'{base_name}_coordinates.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"已保存坐标分布图至 {save_path}")
        
        plt.show(block=True)

def main():
    """
    主函数，运行点标记工具
    """
    image_path = '../datasets/6250位置.png'
    output_file = '../datasets/6250_Max_position.csv'
    start_number = 1  # 从编号1开始
    
    print(f"开始标记工具，起始编号: {start_number}")
    marker = PointMarker(image_path, start_number=start_number)
    plt.show(block=True)
    marker.save_coordinates(output_file)

if __name__ == '__main__':
    main()