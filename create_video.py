import cv2
import numpy as np

def create_10s_test_video(path, fps=25, duration=10, width=1280, height=720):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, fps, (width, height))
    total_frames = fps * duration

    for i in range(total_frames):
        # Generate some pattern so it's not empty
        frame = np.full((height, width, 3), (i % 255, (i * 2) % 255, (i * 3) % 255), dtype=np.uint8)
        out.write(frame)

    out.release()
    print(f"Created {path}")

if __name__ == '__main__':
    create_10s_test_video('tests/fixtures/real_test_video.mp4')
