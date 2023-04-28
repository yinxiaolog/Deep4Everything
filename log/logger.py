import logging.handlers
import yaml

config_path = './config.yml'


class Logger:
    logger = None
    file = None
    desc = None

    @staticmethod
    def get_log_config():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            Logger.file = config['log']['file']
            Logger.desc = config['log']['desc']

    @staticmethod
    def get_logger():
        if Logger.logger is not None:
            return Logger.logger

        Logger.get_log_config()
        Logger.logger = logging.Logger("Logger")
        if 'file' in Logger.desc:
            file_handler = logging.handlers.RotatingFileHandler(filename=Logger.file)
            Logger.logger.addHandler(file_handler)
        if 'console' in Logger.desc:
            console_handler = logging.StreamHandler()
            Logger.logger.addHandler(console_handler)

        return Logger.logger
