import logging
import sys
from colorama import Fore, Style, init

init(autoreset=True)  # Reset colors automatically after each print

LEVEL_COLORS = {
    logging.DEBUG: Fore.CYAN,
    logging.INFO: Fore.GREEN,
    logging.WARNING: Fore.YELLOW,
    logging.ERROR: Fore.RED,
    logging.CRITICAL: Fore.MAGENTA,
}


class ColorFormatter(logging.Formatter):
    def format(self, record):
        color = LEVEL_COLORS.get(record.levelno, "")
        msg = super().format(record)
        return f"{color}{msg}{Style.RESET_ALL}"


class AppLogger:
    def __init__(self, name: str = "InfoSynth", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if not self.logger.handlers:
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = ColorFormatter(
                "[%(asctime)s] [%(levelname)s] - %(message)s", "%Y-%m-%d %H:%M:%S"
            )
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

    def get_logger(self):
        return self.logger
