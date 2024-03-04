import cv2
import numpy as np
from skimage.metrics import structural_similarity
from tqdm import tqdm
import logging as log

log.basicConfig(level=log.DEBUG)


class VideoChangeDetector:
    FG_MASK_TYPES = {
        "MOG2": cv2.createBackgroundSubtractorMOG2,
        "KNN": cv2.createBackgroundSubtractorKNN,
    }

    def __init__(
        self, video_path, object_detection_model="YOLOv5"
    ):  # Flexibility to choose a detection model
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.reference_frame = None
        self.object_detection_model = self._load_object_detection_model(
            object_detection_model
        )
        self.cycle_times = []

    def _load_object_detection_model(self, model_name):
        # Load the specified object detection model (YOLOv5, SSD, etc.)
        return model_name  # Placeholder for now

    def _extract_first_frame(self):
        # Read the first frame and store it as self.reference_frame
        ret, self.reference_frame = self.cap.read()
        if not ret:
            raise ValueError("Error reading the first frame")

    def _detect_objects(self, frame):
        # TODO: Use self.object_detection_model to detect objects in the frame
        # self.object_detection_model.detect(frame)
        return frame  # Placeholder for now

    def _calculate_difference(self, frame, fg_mask_type="MOG2"):
        # Implement frame differencing or background subtraction with the reference frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Background subtraction
        if self.reference_frame is None:  # Handle the case without a reference frame
            self.reference_frame = gray_frame
            return np.zeros_like(gray_frame)  # No difference on the first frame

        if fg_mask_type in self.FG_MASK_TYPES:
            fg_mask = self.FG_MASK_TYPES[fg_mask_type]().apply(gray_frame)
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

    def _check_similarity(self, frame):
        # Calculate similarity metrics (SSIM, MSE) between frame and reference_frame
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_reference = cv2.cvtColor(self.reference_frame, cv2.COLOR_BGR2GRAY)

        (score, diff) = structural_similarity(gray_reference, gray_frame, full=True)
        similarity_condition_met = score >= 0.95  # Example threshold, adjust as needed

        return similarity_condition_met

    def detect_cycles(self):
        self._extract_first_frame()

        cycle_count = 0
        self.cycle_times.append({"start": "00:00:00", "end": "00:00:00"})

        with tqdm(total=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))) as pbar:
            while True:
                pbar.update(1)
                # TODO: Read subsequent frames from the video
                ret, current_frame = self.cap.read()
                if not ret:
                    log.info("End of video")
                    break

                # TODO: Detect objects using _detect_objects()
                objects_detected = self._detect_objects(current_frame)

                # TODO: Skip frames without objects detected for efficiency
                if not objects_detected:
                    continue

                # TODO: Calculate difference with reference frame using _calculate_difference()
                difference_frame = self._calculate_difference(current_frame)
                log.info("Difference frame:", difference_frame.shape)

                # TODO: Detect change using _detect_change()
                changed_detected = self._detect_change(difference_frame)
                log.info("Change detected:", changed_detected)

                # TODO: Check similarity using _check_similarity()
                similarity_condition_met = self._check_similarity(current_frame)
                if changed_detected and similarity_condition_met:
                    current_time_frame = self.cap.get(cv2.CAP_PROP_POS_MSEC)
                    log.info("Change detected at:", current_time_frame)
                    self.cycle_times[cycle_count]["end"] = current_time_frame
                    cycle_count += 1

    def get_cycle_times(self):
        return self.cycle_times


# Example Usage:
detector = VideoChangeDetector("video_data/video_sample_crop_01.mp4")
detector.detect_cycles()
cycle_times = detector.get_cycle_times()

log.info("Cycle times: \n", cycle_times)
