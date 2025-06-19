import logging


class Monitor:
    """Basic performance monitor that logs metrics to a file."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("monitor")
        if not self.logger.handlers:
            handler = logging.FileHandler("monitor.log")
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)

    def log_metric(self, name: str, value: float):
        self.logger.info("%s: %s", name, value)


__all__ = ["Monitor"]
