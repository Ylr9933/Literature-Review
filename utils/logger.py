# study_utils/log_utils.py

import logging
import os

class Logger:
    def __init__(self, save_dir="logs", fname="app.log"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        log_path = os.path.join(save_dir, fname)

        # 配置日志格式
        self.logger = logging.getLogger("StreamlitApp")
        self.logger.setLevel(logging.DEBUG)

        # 避免重复添加 handler
        if not self.logger.handlers:
            file_handler = logging.FileHandler(log_path, mode='a', encoding='utf-8')
            console_handler = logging.StreamHandler()

            formatter = logging.Formatter(f"[%(asctime)s] %(levelname)s: %(message)s")
            file_handler.setFormatter(formatter)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)

    def log(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)

    def debug(self, message):
        self.logger.debug(message)

    def warning(self, message):
        self.logger.warning(message)