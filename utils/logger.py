"""
"""
import logging

class Logger():
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    def __call__(self, log_path):
        """
        """
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            # Logging to a file
            file_handler = logging.FileHandler(log_path)
            file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
            logger.addHandler(file_handler)

            # Logging to console
            stream_handler = logging.StreamHandler()
            stream_handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(stream_handler)
