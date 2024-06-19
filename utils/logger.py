import os
import logging


class Logger:
    """
    Logger class.


    Parameters
    ----------
    :param name: str
        The name of the logger.

    :param log_to_console: bool, default=False
        Whether to log to the console.
    :param log_filename: str, default=None
        The name of the log file.
    :param log_level: str, default="INFO"
        The level of logging to use.
        Options are "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL".
    :param overwrite_file: bool, default=True
        Whether to overwrite the log file if it already exists.
    """

    def __init__(
        self,
        name: str,
        log_to_console: bool = False,
        log_filename: str = None,
        log_level: str = "INFO",
        overwrite_file=True,
    ):
        self.log_to_console = log_to_console
        self.log_filename = log_filename

        # Create or retrieve a logger with the given name
        self.logger = logging.getLogger(name)
        # Set the logging level
        self.logger.setLevel(getattr(logging, log_level.upper(), "INFO"))

        # Clear any existing handlers
        self.logger.handlers.clear()

        # Create a log format
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # If a log filename is provided, add a file handler
        if log_filename:
            file_mode = (
                "w" if overwrite_file else "a"
            )  # Overwrite or append to the file
            file_handler = DeferredFileHandler(log_filename, mode=file_mode)
            file_handler.setFormatter(log_format)
            self.logger.addHandler(file_handler)

        # If log to console is True, add a console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(log_format)
            self.logger.addHandler(console_handler)

    def info(self, message):
        self.logger.info(message)

    def error(self, message):
        self.logger.error(message)


class DeferredFileHandler(logging.FileHandler):
    """
    File handler that creates the file only when the first log message is emitted.
    """

    def __init__(self, filename, mode="a", encoding=None, delay=True):
        self._filename = filename
        self._mode = mode
        self._encoding = encoding
        self._delay = delay
        super().__init__(filename, mode, encoding, delay)

    def emit(self, record):
        if self.stream is None:  # File not created yet
            if not os.path.exists(os.path.dirname(self._filename)) and os.path.dirname(
                self._filename
            ):
                os.makedirs(os.path.dirname(self._filename))  # Ensure directory exists
            self.stream = self._open()  # Create file here
        super().emit(record)
