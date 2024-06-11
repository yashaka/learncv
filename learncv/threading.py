from functools import reduce
from typing import Callable, TypeVar, Any


T = TypeVar("T")


def thread_last(arg, *iterable_of_fn_or_tuple):
    """Thread the first argument as last argument
    through a sequence of functions or tuples of functions
    with arguments in order to get rid of nested calls.

    Examples:
        >>> from cv_r_d_course.threading import thread_last, map_with
        >>> import re
        >>> result = thread_last(
        >>>     ['_have.special_number_', 10],
        >>>     map_with(str),
        >>>     ''.join,
        >>>     (re.sub, '(^_+|_+$)', ''),
        >>>     (re.sub, '_+', ' '),
        >>>     (re.sub, '(\\w)\\.(\\w)', '\\1 \\2'),
        >>>     str.split,
        >>> )
        >>> assert result == ['have', 'special', 'number', '10']
    """
    return (
        reduce(
            lambda acc, fn_or_tuple: (
                fn_or_tuple(acc)
                if callable(fn_or_tuple)
                else fn_or_tuple[0](*fn_or_tuple[1:], acc)
            ),
            iterable_of_fn_or_tuple,
            arg,
        )
        if iterable_of_fn_or_tuple
        else arg
    )


def thread_first(arg, *iterable_of_fn_or_tuple):
    """Thread the first argument as first argument
    through a sequence of functions or tuples of functions
    with arguments in order to get rid of nested calls.
    """
    return (
        reduce(
            lambda acc, fn_or_tuple: (
                fn_or_tuple(acc)
                if callable(fn_or_tuple)
                else fn_or_tuple[0](acc, *fn_or_tuple[1:])
            ),
            iterable_of_fn_or_tuple,
            arg,
        )
        if iterable_of_fn_or_tuple
        else arg
    )


def map_with(fn: Callable):
    """Curried non-lazy version of map function.

    It is curried for easier usage in higher order functions like thread_last.
    It is non-lazy, i.e. with applied list to result, because for CV tasks
    we usually need to apply something to all list items after map application.
    """
    return lambda *iterables: list(map(fn, *iterables))


def filter_with(predicate: Callable):
    """Curried non-lazy version of filter function.

    It is curried for easier usage in higher order functions like thread_last.
    It is non-lazy, i.e. with applied list to result, because for CV tasks
    we usually need to apply something to all list items after filter application.
    """
    return lambda iterable: list(filter(predicate, iterable))


def do(function: Callable[[T], Any]) -> Callable[[T], T]:
    def func(arg: T) -> T:
        function(arg)
        return arg

    return func


# TODO: consider refactoring to utilize the do function defined above
def setitem(iterable, slice_or_index, value):
    iterable.__setitem__(slice_or_index, value)
    return iterable


def setitem_by(slice_or_index, value):
    def fn(iterable):
        iterable.__setitem__(slice_or_index, value)
        return iterable

    return fn
