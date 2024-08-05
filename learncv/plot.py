from typing import Tuple, List, Any, Callable, TypeVar, Iterable
from matplotlib import pyplot as plt
import numpy as np

TIndex = TypeVar('TIndex', bound=int)


# todo: consider refactoring to accept `data: Tuple[TImages, TLabels] | TData``
#       over `images, labels`, with additional `__getitem__`-like strategies
#       for both images and labels
#       by default equal to simply `lambda idx: iterable[idx]` where
#       `iterable` is either `images` or `labels`
def subset(
    images: List[np.ndarray] | Callable[[TIndex], np.ndarray],
    labels: List[Any] | Callable[[TIndex], Any],
    number: Tuple[int, int] | int = (2, 5),
    among: int | Iterable | None = None,
    cmap=None,
    vmin=0,
    vmax=255,
):
    if among is None and callable(images):
        raise AttributeError("among is required when images is a callable")
    # todo: refactor to make mypy happy (currently the algorithm is concise
    #       but not correct-typing-proved and so prone to errors on future-refactoring)
    among = among if isinstance(among, int) else len(images if among is None else among)
    images = images if callable(images) else lambda idx: images[idx]
    labels = labels if callable(labels) else lambda idx: labels[idx]

    rows, cols = (
        number
        if isinstance(number, tuple)
        else ((number // 5, 5) if number % 5 == 0 else (number // 5 + 1, 5))
    )

    for cnt, idx in enumerate(
        np.random.randint(0, among, rows * cols if isinstance(number, tuple) else number)
    ):
        plt.subplot(rows, cols, cnt + 1)
        plt.imshow(images(idx), cmap=cmap, vmin=vmin, vmax=vmax)
        plt.title(labels(idx)), plt.axis(False)


def training_history(training, /):
    history = training.history
    epochs = range(len(history['loss']))
    plt.plot(epochs, history['loss'], '.-'), plt.grid(True)
    plt.xlabel('epoch'), plt.ylabel('loss')
