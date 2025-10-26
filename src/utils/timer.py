import time
from contextlib import contextmanager

@contextmanager
def timer():
    t0 = time.time()
    yield lambda: time.time() - t0
