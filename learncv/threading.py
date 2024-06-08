from functools import reduce
from typing import Callable


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


def map_with(fn: Callable):
    """Curried non-lazy version of map function.

    It is curried to for easier usage in higher order functions like thread_last.
    It is non-lazy, i.e. with applied list to result, because for CV tasks
    we usually need to apply something to all list items after map application.
    """
    return lambda *iterables: list(map(fn, *iterables))
