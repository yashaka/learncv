import cv2
from matplotlib import pyplot as plt
import numpy as np
from typing import Literal, Dict, Callable
from itertools import count


def fast_forward(video, frames=24, color_space=cv2.COLOR_BGR2RGB):
    for _ in range(frames):
        ret, frame = video.read()
        if not ret:
            break
    return cv2.cvtColor(frame, color_space) if color_space else frame


def cv2_show_and_pressed_q(img, title=None, bbox=None):
    if bbox:
        img = img.copy()
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imshow(title or 'Image', img)
    return cv2.waitKey(1) & 0xFF == ord('q')


def plt_show(img, title=None, bbox=None):
    if bbox:
        img = img.copy()
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show(), plt.draw()
    # plt.waitforbuttonpress(pause)
    plt.clf()


def fast_forward_and_show_next(video, frames=24):
    fast_forward(video, frames)
    ret, frame = video.read()
    if not ret:
        return
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt_show(frame_rgb, f'Frame {frames+1}')
    return frame_rgb


TrackerType = Literal['MIL', 'KCF', 'GOTURN', 'CSRT']
trackers: Dict[TrackerType, Callable] = {
    'MIL': cv2.TrackerMIL_create,
    'KCF': cv2.TrackerKCF_create,
    'GOTURN': cv2.TrackerGOTURN_create,  # This does not work :(
    'CSRT': cv2.TrackerCSRT_create,
}


def track(
    video,
    frame: np.ndarray | int,
    bbox,
    stop: Dict[Literal['at', 'after'], int] = {},
    type_: TrackerType = 'CSRT',
    debug=False,
):
    tracker = trackers[type_]()

    template = fast_forward(video, frame) if isinstance(frame, int) else frame
    stop_after = stop.get(
        'after', stop.get('at', 0) - (frame if isinstance(frame, int) else 0)
    )
    if debug:
        print(f'Stop after {stop_after} frames')
    counter = count(1) if stop_after > 0 else None

    template = (
        cv2.cvtColor(template, cv2.COLOR_RGB2GRAY) if type_ not in ['KCF'] else template
    )
    ok = tracker.init(template, bbox)

    while True:
        ret, frame_bgr = video.read()
        if not ret:
            break

        new_template = (
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            if type_ not in ['KCF']
            else frame_bgr
        )
        ok, bbox = tracker.update(new_template)
        if debug:
            print(ok, bbox)

        if cv2_show_and_pressed_q(frame_bgr, 'Tracking', bbox):
            break

        if stop_after > 0 and counter.__next__() >= stop_after:
            break
