import cv2
from app.pipeline.frame_extractor import extract_frames

def test_it():
    gen = extract_frames('tests/fixtures/real_test_video.mp4', fps_target=5)
    _ = next(gen)
    cap = gen.gi_frame.f_locals['cap']
    print("Opened before:", cap.isOpened())
    list(gen)
    print("Opened after:", cap.isOpened())

test_it()
