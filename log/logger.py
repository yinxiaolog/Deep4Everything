import logging.handlers
import os

import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Logger:
    __logger = None
    __file = None
    __desc = None
    __log_base_dir = None
    model_name = None
    time = None
    log_fmt = logging.Formatter("[%(asctime)s][%(levelname)s][func:%(funcName)s]: %(message)s")

    @staticmethod
    def set(model_name, time):
        Logger.model_name = model_name
        Logger.time = time
        Logger.__get_log_config(model_name)
        if 'file' in Logger.__desc and Logger.model_name is not None and Logger.time is not None:
            log_path = os.path.join(Logger.__log_base_dir,
                                    model_name + '_' + time)
            if not os.path.exists(log_path):
                os.makedirs(log_path)

            log_file = os.path.join(log_path, 'train.log')
            if not os.path.exists(log_file):
                os.mknod(log_file)

            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_file)
            file_handler.setFormatter(Logger.log_fmt)
            Logger.__logger.addHandler(file_handler)
        if 'console' in Logger.__desc:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(Logger.log_fmt)
            Logger.__logger.addHandler(console_handler)

    @staticmethod
    def __get_log_config(model_name):
        config_path = os.path.join(base_dir, 'modelsConfig', model_name + '.yml')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            Logger.__file = config['log']['file']
            Logger.__desc = config['log']['desc']
            Logger.__log_base_dir = config['log']['log_base_dir']

    @staticmethod
    def get_logger():
        if Logger.__logger is not None:
            return Logger.__logger
        Logger.__logger = logging.Logger("Logger")
        return Logger.__logger
