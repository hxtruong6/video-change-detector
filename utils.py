import cv2
import numpy as np


def crop_by_percent(video_path, crop_percent_width, crop_percent_height, output_path):
    cap = cv2.VideoCapture(video_path)

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Calculate target dimensions based on percentages
    crop_width = int(original_width * crop_percent_width)
    crop_height = int(original_height * crop_percent_height)

    # Calculate starting coordinates for the crop
    x_start = (original_width - crop_width) // 2
    y_start = (original_height - crop_height) // 2

    # Define output video properties
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4 codec
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_path, fourcc, fps, (crop_width, crop_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame
        cropped_frame = frame[
            y_start : y_start + crop_height, x_start : x_start + crop_width
        ]

        # Write the cropped frame to the output video
        out.write(cropped_frame)

        if cv2.waitKey(1) == ord("q"):  # Press 'q' to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Example usage
video_path = "video_data/video_sample_crop_01.mp4"
crop_percent_width = 0.5  # Crop to 50% of the original width
crop_percent_height = 0.7  # Crop to 70% of the original height
output_path = "output_cropped.mp4"

crop_by_percent(video_path, crop_percent_width, crop_percent_height, output_path)
