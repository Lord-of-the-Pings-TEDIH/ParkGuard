import os
import cv2
import numpy as np
import pytest
import tempfile
from app.pipeline.frame_extractor import extract_frames, get_video_info


def create_test_video(path, fps, num_frames, width=100, height=100):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))

    for i in range(num_frames):
        # Create a simple frame: just a single color varying by frame index
        frame = np.full((height, width, 3), (i % 255, (i * 2) % 255, (i * 3) % 255), dtype=np.uint8)
        out.write(frame)

    out.release()
    return path


@pytest.fixture
def test_video(tmp_path):
    video_path = str(tmp_path / "test_video.mp4")
    create_test_video(video_path, fps=30, num_frames=60)
    return video_path


def test_extract_frames_invalid_path():
    with pytest.raises(ValueError, match="Cannot open video: 'nonexistent.mp4'"):
        list(extract_frames("nonexistent.mp4", fps_target=10))


def test_extract_frames_target_fps(test_video):
    # Original video is 30 FPS, target is 10 FPS
    # So we expect roughly 30/10 = 3 frame interval, meaning we take 1 frame every 3 frames
    frames = list(extract_frames(test_video, fps_target=10))

    # We generated 60 frames, so we expect 20 frames (indices: 0, 3, 6, ..., 57)
    assert len(frames) == 20

    expected_indices = list(range(0, 60, 3))
    actual_indices = [f[0] for f in frames]
    assert actual_indices == expected_indices

    # Check PTS calculations
    # fps is 30. pts_ms for frame_index 3 = round(3 / 30 * 1000) = 100
    for idx, (frame_index, pts_ms, frame) in enumerate(frames):
        assert pts_ms == round(frame_index / 30 * 1000)
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (100, 100, 3)


def test_extract_frames_higher_target_fps(test_video):
    # Original video is 30 FPS, target is 60 FPS
    # interval should be max(1, round(30 / 60)) = max(1, 1) = 1
    # Thus, it returns all frames
    frames = list(extract_frames(test_video, fps_target=60))

    assert len(frames) == 60

    expected_indices = list(range(60))
    actual_indices = [f[0] for f in frames]
    assert actual_indices == expected_indices


def test_get_video_info(test_video):
    info = get_video_info(test_video)
    assert info['fps'] == 30.0
    assert info['total_frames'] == 60
    assert info['width'] == 100
    assert info['height'] == 100
    assert info['duration_s'] == 2.0


def test_real_video_extraction():
    video_path = "tests/fixtures/real_test_video.mp4"
    gen = extract_frames(video_path, fps_target=5)
    
    # Get the first frame to initialize the generator and open the cap
    first_yield = next(gen)
    
    # Introspect local variables to get the 'cap' object
    cap = gen.gi_frame.f_locals['cap']
    
    # Consume the rest of the frames
    frames = [first_yield] + list(gen)
    
    # 1. Test that fps_target=5 yields ~50 frames from a 10s video (±2 tolerance)
    assert 48 <= len(frames) <= 52, f"Expected ~50 frames, got {len(frames)}"
    
    # 2. Test that pts_ms increases monotonically
    pts_list = [f[1] for f in frames]
    assert all(pts_list[i] < pts_list[i+1] for i in range(len(pts_list)-1))
    
    # 3. Test that frame.shape[2] == 3 (BGR)
    for _, _, frame in frames:
        assert frame.shape[2] == 3
        
    # 4. Test that VideoCapture is closed after iteration
    assert cap.isOpened() == False

