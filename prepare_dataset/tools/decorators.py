# pylint: disable = unable to import:E0401, c0411, c0412, w0611, w0621, c0103
import random
import time
from atexit import register
from dataclasses import dataclass
# from dataclasses import asdict, astuple, field
from functools import lru_cache, singledispatch, wraps

import requests


def logger(function):
    """logger is a function that takes a function as input and returns a function as output."""

    @wraps(
        function
    )  # updates the wrapper function to look like the original function and inherit its name and properties
    def wrapper(*args, **kwargs):
        """wrapper documentation"""
        print(f"---- {function.__name__}: start ----")
        output = function(*args, **kwargs)
        print(f"---- {function.__name__}: end ----")
        return output

    return wrapper


def repeat(number_of_times):
    """causes a function to be called multiple times in a row.
    This can be useful for debugging purposes, stress tests, or automating the repetition of multiple tasks.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for _ in range(number_of_times):
                func(*args, **kwargs)

        return wrapper

    return decorate


def timeit(func):
    """measures the execution time of a function and prints the result: this serves as debugging or monitoring."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.6f} seconds to complete")
        return result

    return wrapper


def retry(num_retries, exception_to_check, sleep_time=0):
    """
    Decorator that retries the execution of a function if it raises a specific exception.
    """

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(1, num_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    print(f"{func.__name__} raised {e.__class__.__name__}. Retrying...")
                    if i < num_retries:
                        time.sleep(sleep_time)
            # Raise the exception if the function was not successful after the specified number of retries
            # raise e

        return wrapper

    return decorate


def countcall(func):
    """counts the number of times a function has been called."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        result = func(*args, **kwargs)
        print(f"{func.__name__} has been called {wrapper.count} times")
        return result

    wrapper.count = 0
    return wrapper


def rate_limited(max_per_second):
    """limits the rate at which a function can be called, by sleeping an amount of time if the function is called too frequently."""
    min_interval = 1.0 / float(max_per_second)

    def decorate(func):
        last_time_called = [0.0]

        @wraps(func)
        def rate_limited_function(*args, **kargs):
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kargs)
            last_time_called[0] = time.perf_counter()
            return ret

        return rate_limited_function

    return decorate


@dataclass
class Person:
    first_name: str
    last_name: str
    age: int
    job: str

    def __eq__(self, other):
        if isinstance(other, Person):
            return self.age == other.age
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Person):
            return self.age < other.age
        return NotImplemented


@logger
def some_function(text):
    print(text)


@logger
def add_two_numbers(a, b):
    """this is functin to add two numbers and return the output"""
    return a + b


@lru_cache(maxsize=None)  # It caches the return values of a function
def heavy_processing(n):
    sleep_time = n + random.random()
    time.sleep(sleep_time)


@repeat(5)
def dummy():
    print("test repeat decorator.")


@timeit
def process_data():
    time.sleep(1)


@retry(num_retries=3, exception_to_check=ValueError, sleep_time=1)
def random_value():
    value = random.randint(1, 5)
    if value == 3:
        raise ValueError("Value cannot be 3")
    return value


@countcall
def process_data():
    pass


@rate_limited(15)
def call_api(url):
    response = requests.get(url)

    if response.status_code != 200:
        raise Exception("response: {}".format(response.status_code))
    return response


class Movie:
    def __init__(self, r):
        self._rating = r

    @property  # used to define class properties which are essentially getter, setter, and deleter methods for a class instance attribute
    def rating(self):
        return self._rating

    @rating.setter
    def rating(self, r):
        if 0 <= r <= 5:
            self._rating = r
        else:
            raise ValueError("The movie rating must be between 0 and 5!")


@register
def terminate():
    """If your Python script accidentally terminates and you still want to perform some
    tasks to save your work, perform cleanup or print a message
    """
    # do something such as save, cleanup or print a message
    print("saved the work with the accident termination")


@singledispatch  # allows a function to have different implementations for different types of arguments.
def fun(arg):
    print("Called with a single argument")


@fun.register(int)
def _(arg):
    print(f"Called with an integer: {arg}")


@fun.register(list)
def _(arg):
    print(f"Called with a list: {arg}")


# some_function("first test")
# add_two_numbers(3,6)
# print(some_function.__name__)
# print(some_function.__doc__)
# print(add_two_numbers.__name__)
# print(add_two_numbers.__doc__)
#################################################################################
# time.time
# heavy_processing(0) # first time
# heavy_processing(0) # second time
# heavy_processing(0) # third time
#################################################################################
# dummy()
#################################################################################
# process_data()
#################################################################################
# print(random_value())
#################################################################################
# for i in range(3):
#     process_data()

#################################################################################
# print(call_api(r'https://www.google.com/'))

######################################################################
# john = Person(first_name="John",
#             last_name="Doe",
#             age=30,
#             job="doctor",)

# anne = Person(first_name="Anne",
#             last_name="Smith",
#             age=40,
#             job="software engineer",)

# print(john == anne)
# print(anne > john)
# print(asdict(anne))
# print(astuple(anne))
#################################################################################
# batman = Movie(2.5)
# print(batman.rating)
# batman.rating = 4
# print(batman.rating)
# batman.rating = 10
# print(batman.rating)
#################################################################################
# # test terminate decorator, press ctrl+c to stop while loop and then terminate run.
# while True:
#     print("Hello")


#################################################################################
# fun(1)  # Prints "Called with an integer"
# fun([1, 2, 3])  # Prints "Called with a list"
#################################################################################
