from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timed(records, label):
    start = perf_counter()
    try:
        yield
    finally:
        records[label] = records.get(label, 0.) + perf_counter() - start
