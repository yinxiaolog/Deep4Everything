import logging.handlers
import yaml
import os

config_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(config_dir, 'config.yml')


class Logger:
    __logger = None
    __file = None
    __desc = None

    @staticmethod
    def __get_log_config():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            Logger.__file = config['log']['file']
            Logger.__desc = config['log']['desc']

    @staticmethod
    def get_logger():
        if Logger.__logger is not None:
            return Logger.__logger

        Logger.__get_log_config()
        Logger.__logger = logging.Logger("Logger")
        log_fmt = logging.Formatter("[%(asctime)s][%(levelname)s][func:%(funcName)s]: %(message)s")
        if 'file' in Logger.__desc:
            file_handler = logging.handlers.RotatingFileHandler(filename=Logger.__file)
            file_handler.setFormatter(log_fmt)
            Logger.__logger.addHandler(file_handler)
        if 'console' in Logger.__desc:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_fmt)
            Logger.__logger.addHandler(console_handler)

        return Logger.__logger
