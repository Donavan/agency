# logging_utils.py
import logging
import threading
import os
from pathlib import Path
from agent_c.util import debugger_is_active


class LoggingManager:
    _instance = None
    _debug_event = None

    def __new__(cls, logger_name: str, log_dir: str = "logs"):
        """
        Ensure we only create one debug event across all instances.

        Args:
            logger_name (str): Name for the logger instance
            log_dir (str): Directory for log files. Defaults to "logs"
        """
        if cls._instance is None:
            cls._instance = super(LoggingManager, cls).__new__(cls)
            cls._debug_event = threading.Event()
            if debugger_is_active():
                cls._debug_event.set()
        return cls._instance

    def __init__(self, logger_name: str, log_dir: str = "logs"):
        """
        Initialize the logging manager.

        Args:
            logger_name (str): Name for the logger instance
            log_dir (str): Directory for log files. Defaults to "logs"
        """
        self.logger_name = logger_name
        self.log_dir = log_dir
        self.logger = self.__setup_logging()

    def __ensure_log_directory(self):
        """Create log directory if it doesn't exist."""
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)

    def __setup_logging(self) -> logging.Logger:
        """
        Set up a logger instance with specified logging configurations.

        Returns:
            logging.Logger: Configured logger instance
        """
        logger = logging.getLogger(self.logger_name)

        # Only add handlers if they don't exist
        if not logger.handlers:
            # Create formatter
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)

            if hasattr(console_handler.stream, 'reconfigure'):
                console_handler.stream.reconfigure(encoding='utf-8')

            # File handler
            self.__ensure_log_directory()
            file_handler = logging.FileHandler(
                os.path.join(self.log_dir, f"{self.logger_name.replace('.', '_')}.log"),
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)

            # Add both handlers
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        other_loggers = [
            'httpx', 'LiteLLM', 'openai', 'httpcore', 'websockets',
            'speechmatics', 'asyncio', 'linkedin_api', 'httpcore',
            'urllib3', 'gradio'
        ]

        debug_other_loggers = [
            'agent_c_core', 'agent_c_tools', 'agent_c_tools.tools',
            'agent_c_demo', 'agent_c_demo.tools',
            'agent_c_reference_apps.react_fastapi.backend'
        ]

        if self._debug_event.is_set():
            logger.setLevel(logging.DEBUG)
            for log in debug_other_loggers:
                debug_logger = logging.getLogger(log)
                debug_logger.setLevel(logging.DEBUG)
                # Add file handler to debug loggers if they don't have one
                if not any(isinstance(h, logging.FileHandler) for h in debug_logger.handlers):
                    debug_file_handler = logging.FileHandler(
                        os.path.join(self.log_dir, f"{log.replace('.', '_')}.log"),
                        encoding='utf-8'
                    )
                    debug_file_handler.setFormatter(formatter)
                    debug_logger.addHandler(debug_file_handler)
        else:
            logger.setLevel(logging.WARN)

        for log in other_loggers:
            logging.getLogger(log).setLevel(logging.WARN)

        return logger

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: The configured logger
        """
        return self.logger

    @classmethod
    def get_debug_event(cls) -> threading.Event:
        """
        Get the shared debug event instance.

        Returns:
            threading.Event: The debug event
        """
        if cls._debug_event is None:
            cls._debug_event = threading.Event()
            if debugger_is_active():
                cls._debug_event.set()
        return cls._debug_event