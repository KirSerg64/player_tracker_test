import cv2

def frame_generator(video_path, frame_ids):
    """
    Yields frames from a video file by the given frame IDs.

    Args:
        video_path (str): Path to the video file.
        frame_ids (Iterable[int]): Sorted iterable of frame indices to extract.

    Yields:
        (int, np.ndarray): Tuple of (frame_id, frame_image)
    """
    frame_ids = iter(sorted(frame_ids))
    next_id = next(frame_ids, None)
    if next_id is None:
        return

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while cap.isOpened() and next_id is not None:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == next_id:
            yield frame_idx, frame
            next_id = next(frame_ids, None)
        frame_idx += 1
    cap.release()