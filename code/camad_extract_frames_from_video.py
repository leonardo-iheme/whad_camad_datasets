import cv2
import os


def extract_frames(video_path: str, save_dir: str, start_time: float, interval: float, extract_length: float) -> None:
    """
    Extracts frames from a video and saves them to a specified directory.
    :param video_path: Path to the video file
    :param save_dir: Directory to save the frames
    :param start_time: Start time in minutes
    :param interval: Interval between frames in minutes
    :param extract_length: Length of the video to extract in minutes
    :return: None
    """
    try:
        # Check if the video file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Create the save directory if it does not exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise IOError(f"Error opening video file: {video_path}")

        # Get the video framerate
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Calculate the actual time per frame and the interval in frames
        # 10 minutes corresponds to 20 frames in actual time, 1 minute = 2 frames
        frames_per_minute = 2
        interval_in_frames = int(interval * frames_per_minute)
        start_frame = int(start_time * frames_per_minute) - 1
        end_frame = start_frame + int(extract_length * frames_per_minute) - 1

        current_frame = start_frame + 1

        # Set the starting frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        while current_frame <= end_frame:
            ret, frame = cap.read()

            if not ret:
                print(f"Frame read failed at frame {current_frame}. Stopping extraction.")
                break

            # Save the current frame
            frame_filename = os.path.join(save_dir, f"frame_{current_frame:03d}.jpg")
            cv2.imwrite(frame_filename, frame)
            print(f"Saved frame {current_frame} to {frame_filename}")

            # Move to the next frame based on the interval
            current_frame += interval_in_frames
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        # Release the video capture object
        cap.release()
        print("Frame extraction completed successfully.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    extract_frames('../data/camad/videos/exp3_glassrawmatrixconfluent231liveimaging8aug.20fps.blackaddedfor30sec.avi',
                   '../results',
                   1.5, 10, 60)
