import base64
import io
import matplotlib.pyplot as plt
from typing import Optional

def save_plot_as_base64(fig) -> str:
    """
    将matplotlib图表转换为base64编码的字符串
    
    参数
    ----------
    fig : matplotlib.figure.Figure
        matplotlib图表对象
        
    返回
    -------
    str
        完整的data URL格式的图片字符串
    """
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    buffer.close()
    plt.close(fig)  # 关闭图表以释放内存
    return f"data:image/png;base64,{image_base64}"  # 返回完整的data URL格式

def validate_excel_file(file_path: str, required_sheet: str = 'dF') -> bool:
    """
    验证Excel文件是否包含必需的工作表
    
    参数
    ----------
    file_path : str
        Excel文件路径
    required_sheet : str
        必需的工作表名称，默认为'dF'
        
    返回
    -------
    bool
        如果文件有效返回True，否则返回False
    """
    try:
        import pandas as pd
        excel_file = pd.ExcelFile(file_path)
        return required_sheet in excel_file.sheet_names
    except Exception:
        return False

def format_file_size(size_bytes: int) -> str:
    """
    格式化文件大小显示
    
    参数
    ----------
    size_bytes : int
        文件大小（字节）
        
    返回
    -------
    str
        格式化的文件大小字符串
    """
    if size_bytes == 0:
        return "0B"
    size_names = ["B", "KB", "MB", "GB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    return f"{size_bytes:.1f}{size_names[i]}"

def create_error_response(message: str, status_code: int = 500) -> dict:
    """
    创建标准化的错误响应
    
    参数
    ----------
    message : str
        错误消息
    status_code : int
        HTTP状态码，默认为500
        
    返回
    -------
    dict
        错误响应字典
    """
    return {
        "success": False,
        "error": message,
        "status_code": status_code
    }

def create_success_response(data: dict, message: str = "操作成功") -> dict:
    """
    创建标准化的成功响应
    
    参数
    ----------
    data : dict
        响应数据
    message : str
        成功消息，默认为"操作成功"
        
    返回
    -------
    dict
        成功响应字典
    """
    response = {
        "success": True,
        "message": message
    }
    response.update(data)
    return response