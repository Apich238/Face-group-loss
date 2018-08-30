
from time import time


class timer:
    def __init__(self):
        self.start = time()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return None

    def now(self):
        return time() - self.start

    def restart(self):
        self.start = time()