import cv2
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm
import logging as log

log.basicConfig(level=log.DEBUG)


def convert_msec_to_hhmmss(msec):
    return (
        str(int(msec / 1000 / 60 / 60))
        + ":"
        + str(int(msec / 1000 / 60) % 60)
        + ":"
        + str(int(msec / 1000) % 60)
    )


class VideoChangeDetector:
    FG_MASK_TYPES = {
        "MOG2": cv2.createBackgroundSubtractorMOG2,
        "KNN": cv2.createBackgroundSubtractorKNN,
    }

    def __init__(
        self, video_path, object_detection_model="YOLOv5", similarity_threshold=0.95
    ):  # Flexibility to choose a detection model
        self.video_path = video_path
        log.info(f"Loading video from: {video_path}")
        self.cap = cv2.VideoCapture(video_path)
        self.reference_frame = None
        self.reference_frame_time = None

        self.object_detection_model = self._load_object_detection_model(
            object_detection_model
        )
        self.cycle_times = []
        self.similarity_threshold = similarity_threshold

    def _load_object_detection_model(self, model_name):
        # Load the specified object detection model (YOLOv5, SSD, etc.)
        return model_name  # Placeholder for now

    def _extract_first_frame(self):
        # Read the first frame and store it as self.reference_frame
        ret, self.reference_frame = self.cap.read()
        self.reference_frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)

        if not ret:
            raise ValueError("Error reading the first frame")

    def _detect_objects(self, frame):
        # TODO: Use self.object_detection_model to detect objects in the frame
        # self.object_detection_model.detect(frame)
        return frame  # Placeholder for now

    def _calculate_difference(self, frame, fg_mask_type="MOG2"):
        # Implement frame differencing or background subtraction with the reference frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_reference = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction
        if self.reference_frame is None:  # Handle the case without a reference frame
            raise ValueError("Reference frame is not set")

        if fg_mask_type in self.FG_MASK_TYPES:
            fg_mask = self.FG_MASK_TYPES[fg_mask_type]().apply(
                gray_frame_reference, gray_frame
            )
        else:
            raise ValueError("Invalid fg_mask_type. Choose from MOG2 or KNN")

        # Optionally, apply thresholding
        _, difference_image = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        return difference_image

    def _detect_change(self, frame):
        # Analyze the difference_image (frame) and apply thresholds to determine change
        change_detected = (
            np.sum(frame) > 0
        )  # Simple check if there are any non-zero pixels
        return change_detected

    def _check_similarity(self, frame, threshold=0.95):
        # Calculate similarity metrics (SSIM, MSE) between frame and reference_frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        (score, diff) = structural_similarity(gray_reference, gray_frame, full=True)
        log.info(
            f"{convert_msec_to_hhmmss(self.reference_frame_time)} - {convert_msec_to_hhmmss(self.cap.get(cv2.CAP_PROP_POS_MSEC))} = SSIM: {round(score, 5)} "
        )
        similarity_condition_met = score >= threshold

        return similarity_condition_met, score

    def _calculate_time_difference(self, start_time, end_time):
        return end_time - start_time

    def detect_cycles(self):
        self._extract_first_frame()

        cycle_count = 0
        self.cycle_times.append({"start": "00:00:00", "end": "00:00:00"})

        with tqdm(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while True:
                pbar.update(1)
                log.info(
                    f"Frame: {self.cap.get(cv2.CAP_PROP_POS_FRAMES)} - {convert_msec_to_hhmmss(self.cap.get(cv2.CAP_PROP_POS_MSEC))}"
                )

                # Skip 30 frames (1s) for efficiency
                self.cap.set(
                    cv2.CAP_PROP_POS_FRAMES, self.cap.get(cv2.CAP_PROP_POS_FRAMES) + 30
                )
                # Read subsequent frames from the video
                ret, current_frame = self.cap.read()

                if not ret:
                    log.info("End of video")
                    break

                # TODO: check object detection later
                # # Detect objects using _detect_objects()
                # objects_detected = self._detect_objects(current_frame)

                # # Skip frames without objects detected for efficiency
                # if objects_detected is None:
                #     continue

                # Calculate difference with reference frame using _calculate_difference()
                # difference_frame = self._calculate_difference(current_frame)
                # log.info("Difference frame:", difference_frame.shape)

                # Detect change using _detect_change()
                # changed_detected = self._detect_change(difference_frame)
                # log.info("Change detected:", changed_detected)

                # Check similarity using _check_similarity()
                similarity_condition_met, similarity_score = self._check_similarity(
                    current_frame, self.similarity_threshold
                )
                if (
                    similarity_condition_met
                    and self._calculate_time_difference(
                        self.reference_frame_time, self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    )
                    >= 4000
                ):
                    current_time_frame = convert_msec_to_hhmmss(
                        self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    )
                    log.info(
                        f"Similarity condition met at {current_time_frame} with SSIM: {similarity_score}"
                    )

                    # Save this frame to file
                    cv2.imwrite(
                        f"result/cycles_{cycle_count}_{convert_msec_to_hhmmss(self.reference_frame_time)}.jpg",
                        self.reference_frame,
                    )
                    cv2.imwrite(
                        f"result/cycles_{cycle_count}_{current_time_frame}.jpg",
                        current_frame,
                    )

                    # log.info(f"Change detected at {current_time_frame}")

                    # Convert the time to HH:MM:SS format
                    self.cycle_times[cycle_count]["end"] = current_time_frame

                    # update the reference frame
                    self.reference_frame = current_frame
                    self.reference_frame_time = self.cap.get(cv2.CAP_PROP_POS_MSEC)

                    self.cycle_times.append(
                        {"start": current_time_frame, "end": "00:00:00"}
                    )

                    cycle_count += 1

    def get_cycle_times(self):
        return self.cycle_times


# Example Usage:
detector = VideoChangeDetector(
    "video_data/video_sample_crop_02.mp4", similarity_threshold=0.88
)

detector.detect_cycles()
cycle_times = detector.get_cycle_times()

log.info(f"Cycle times: {cycle_times}")
