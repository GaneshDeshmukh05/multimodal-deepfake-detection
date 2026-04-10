import cv2
import os

# -----------------------------
# Dataset paths
# -----------------------------
REAL_VIDEO_FOLDER = "dataset/real"
FAKE_VIDEO_FOLDER = "dataset/fake"

REAL_FRAMES_FOLDER = "frames/real_frames"
FAKE_FRAMES_FOLDER = "frames/fake_frames"


# -----------------------------
# Create output directories
# -----------------------------
os.makedirs(REAL_FRAMES_FOLDER, exist_ok=True)
os.makedirs(FAKE_FRAMES_FOLDER, exist_ok=True)


# -----------------------------
# Extract frames from one video
# -----------------------------
def extract_frames(video_path, output_folder, max_frames=30):
    """
    Extract uniformly spaced frames from a video

    Args:
        video_path (str): Path to video file
        output_folder (str): Folder to save frames
        max_frames (int): Number of frames to extract
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"[ERROR] No frames found in: {video_path}")
        return

    # Step for uniform sampling
    step = max(1, total_frames // max_frames)

    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < max_frames:
        ret, frame = cap.read()

        if not ret:
            break

        # Select frames uniformly
        if frame_count % step == 0:
            frame = cv2.resize(frame, (224, 224))

            frame_name = f"frame_{saved_count}.jpg"
            frame_path = os.path.join(output_folder, frame_name)

            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"[INFO] {os.path.basename(video_path)} → {saved_count} frames extracted")


# -----------------------------
# Process all videos in folder
# -----------------------------
def process_videos(input_folder, output_folder):
    """
    Process all videos in a folder
    """

    if not os.path.exists(input_folder):
        print(f"[WARNING] Folder not found: {input_folder}")
        return

    videos = [v for v in os.listdir(input_folder)
              if v.lower().endswith((".mp4", ".avi", ".mov"))]

    print(f"\n[INFO] Found {len(videos)} videos in {input_folder}")

    for video in videos:
        video_path = os.path.join(input_folder, video)
        video_name = os.path.splitext(video)[0]

        # Create separate folder per video (IMPORTANT for LSTM)
        output_path = os.path.join(output_folder, video_name)
        os.makedirs(output_path, exist_ok=True)

        print(f"[PROCESSING] {video}")
        extract_frames(video_path, output_path)


# -----------------------------
# MAIN EXECUTION
# -----------------------------
if __name__ == "__main__":

    print("\n🚀 Starting Frame Extraction...\n")

    print("📂 Processing REAL videos...")
    process_videos(REAL_VIDEO_FOLDER, REAL_FRAMES_FOLDER)

    print("\n📂 Processing FAKE videos...")
    process_videos(FAKE_VIDEO_FOLDER, FAKE_FRAMES_FOLDER)

    print("\n✅ Frame Extraction Completed Successfully!\n")