import cv2
import os


def extract_frames(video_path: str, save_dir: str, start_time: float, interval: float, extract_length: float) -> None:
    """
    Extracts frames from a video and saves them to a specified directory.

    Args:
        video_path (str): Path to the video file.
        save_dir (str): Directory to save the frames.
        start_time (float): Start time in minutes.
        interval (float): Interval between frames in minutes.
        extract_length (float): Length of the video to extract in minutes.

    Returns:
        None
    """
    def validate_video_file(video_path: str) -> None:
        """Check if the video file exists."""
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

    def create_save_directory(save_dir: str) -> None:
        """Create the save directory if it does not exist."""
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def open_video_file(video_path: str):
        """Open the video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Error opening video file: {video_path}")
        return cap

    def calculate_frame_parameters(cap, start_time: float, interval: float, extract_length: float):
        """Calculate frame parameters."""
        frames_per_minute = 2
        interval_in_frames = int(interval * frames_per_minute)
        start_frame = int(start_time * frames_per_minute) - 1
        end_frame = start_frame + int(extract_length * frames_per_minute) - 1
        return start_frame, end_frame, interval_in_frames

    def save_frame(frame, save_dir: str, current_frame: int) -> None:
        """Save the current frame."""
        frame_filename = os.path.join(save_dir, f"frame_{current_frame:03d}.jpg")
        cv2.imwrite(frame_filename, frame)
        print(f"Saved frame {current_frame} to {frame_filename}")

    try:
        validate_video_file(video_path)
        create_save_directory(save_dir)
        cap = open_video_file(video_path)
        start_frame, end_frame, interval_in_frames = calculate_frame_parameters(cap, start_time, interval, extract_length)
        current_frame = start_frame + 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while current_frame <= end_frame:
            ret, frame = cap.read()
            if not ret:
                print(f"Frame read failed at frame {current_frame}. Stopping extraction.")
                break
            save_frame(frame, save_dir, current_frame)
            current_frame += interval_in_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        cap.release()
        print("Frame extraction completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    extract_frames('../data/camad/videos/exp3_glassrawmatrixconfluent231liveimaging8aug.20fps.blackaddedfor30sec.avi',
                   '../results',
                   1.5, 10, 60)
