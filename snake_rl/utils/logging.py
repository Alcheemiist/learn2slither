import logging

def init_logger(name: str = "snake_rl") -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s: %(message)s",
    )
    return logging.getLogger(name)
