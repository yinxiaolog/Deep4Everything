from log.logger import Logger

logger = Logger.get_logger()

if __name__ == '__main__':
    a = 1
    logger.info(f'{a} test')