import logging
import os
import datetime
import sys

def setup_logger(output_dir=None, logger_name="calcium_analysis_app"):
    """
    设置日志记录器，将日志消息输出到控制台和文件。

    参数:
        output_dir (str, optional): 日志文件输出目录。
                                    如果为 None，则默认为 'calcium_app/logs'。
        logger_name (str, optional): 日志记录器的名称。

    返回:
        logging.Logger: 已配置好的日志记录器。
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # 清除现有的处理器，避免在 Streamlit 重复运行时重复添加
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器
    if output_dir is None:
        # 获取当前脚本的路径
        current_script_path = os.path.dirname(os.path.abspath(__file__))
        # 设置默认的日志目录，相对于src目录
        output_dir = os.path.abspath(os.path.join(current_script_path, '..', 'logs'))

    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(output_dir, f"{logger_name}_{timestamp}.log")

    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.info(f"日志记录器设置完成。日志文件位于: {log_file}")
    return logger
