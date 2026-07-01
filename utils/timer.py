import time
from typing import Optional

class Timer:
    """A context manager to measure execution time of code blocks.

    Attributes:
        elapsed_time (float): The measured duration in seconds.
    """
    def __init__(self) -> None:
        self.start_time: Optional[float] = None
        self.elapsed_time: float = 0.0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.start_time is not None:
            self.elapsed_time = time.perf_counter() - self.start_time
