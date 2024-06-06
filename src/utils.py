import time


def time_it(func):
    """A decorator that prints the execution time of the function it decorates."""

    def wrapper(*args, **kwargs):
        start_time = time.time()  # Capture the start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Capture the end time
        print(f"{func.__name__} executed in {end_time - start_time:.4f} seconds")
        return result

    return wrapper
