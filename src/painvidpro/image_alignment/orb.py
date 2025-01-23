"""Base class for the imge alignment."""

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

from painvidpro.image_alignment.base import AlignmentStatus, ImageAlignmentBase
from painvidpro.utils.image_processing import process_input
from painvidpro.video_processing.utils import video_capture_context, video_writer_context


class ImageAlignmentOrb(ImageAlignmentBase):
    def __init__(self):
        """Base class to align images."""
        super().__init__()
        self.set_default_parameters()

    def set_parameters(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Sets the parameters.

        Args:
            params: A dict with the parameters.

        Returns:
            A boolean indicating if the set up was successfull.
            A string indidcating the error if the set up was not successfull.
        """
        self.params.update(params)
        return True, ""

    def set_default_parameters(self):
        self.params = {
            "min_matches": 10,
            "max_homography_deviation": 10,
            "check_keypoint_distance": True,
            "max_keypoint_distance": 7,
            "diasble_tqdm": False,
            "save_failed_alignments": False,  # Saves the frames that failed to align on disk as is
        }

    def _check_keypoint_distance(
        self,
        keypoints1: List[cv2.KeyPoint],
        keypoints2: List[cv2.KeyPoint],
        matches: List[cv2.DMatch],
        max_keypoint_distance: float,
        min_matches: int,
    ) -> Tuple[bool, List[Tuple[float, float]], List[Tuple[float, float]], List[cv2.DMatch]]:
        """
        Extracts location of good matches and filters by distance.

        Args:
            keypoints1: Keypoints from the first image.
            keypoints2: Keypoints from the second image.
            matches: Matches between the keypoints.
            max_keypoint_distance: Maximum allowed distance between matched keypoints.
            min_matches: Minimum number of good matches required.

        Returns:
            - A boolean indicating if there are enough good matches after filtering.
            - List of points from the first image.
            - List of points from the second image.
            - List of good matches.
        """
        points1 = []
        points2 = []
        good_matches = []

        for match in matches:
            pt1 = keypoints1[match.queryIdx].pt
            pt2 = keypoints2[match.trainIdx].pt
            distance = np.linalg.norm(np.array(pt1) - np.array(pt2))
            if distance <= max_keypoint_distance:
                points1.append(pt1)
                points2.append(pt2)
                good_matches.append(match)

        # Check if there are enough good matches after filtering
        if len(good_matches) < min_matches:
            return False, points1, points2, good_matches

        return True, points1, points2, good_matches

    def _align_images(
        self,
        orb: cv2.ORB,
        bf: cv2.BFMatcher,
        keypoints1: List[cv2.KeyPoint],
        descriptors1: List[Any],
        height: int,
        width: int,
        img2: Union[List[np.ndarray], List[str], cv2.VideoCapture],
        min_matches: int = 10,
        max_homography_deviation: int = 10,
        check_keypoint_distance: bool = True,
        max_keypoint_distance: float = 7.0,
    ):
        """
        Aligns the second image to the first image using keypoints and descriptors.

        Args:
            orb: ORB detector for keypoint detection and descriptor computation.
            bf: Brute-force matcher for matching descriptors.
            keypoints1: Keypoints from the first image.
            descriptors1: Descriptors from the first image.
            height: Height of the images.
            width: Width of the images.
            img2: Second image to be aligned.
            min_matches: Minimum number of matches required.
            max_homography_deviation: Maximum allowed deviation for homography.
            check_keypoint_distance: Whether to check the distance between keypoints.
            max_keypoint_distance: Maximum allowed distance between matched keypoints.

        Returns:
            Tuple: Alignment status, aligned image, mask, keypoints from the first image, keypoints from the second image, and matches.
        """
        img2 = process_input(img2)
        # Convert images to grayscale
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
        matches = bf.match(descriptors1, descriptors2)
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)

        # Check if there are enough matches
        if len(matches) < min_matches:
            return (AlignmentStatus.NOT_ENOUGH_MATCHES, None, None, keypoints1, keypoints2, matches)

        if check_keypoint_distance:
            succ, p1, p2, matches = self._check_keypoint_distance(
                keypoints1, keypoints2, matches, max_keypoint_distance, min_matches
            )
            if not succ:
                return (AlignmentStatus.NOT_ENOUGH_GOOD_MATCHES, None, None, keypoints1, keypoints2, matches)

            points1 = np.array(p1, dtype=np.float32)
            points2 = np.array(p2, dtype=np.float32)

        else:
            # Extract location of good matches
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography matrix
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Check if the homography matrix represents a small alignment
        if np.linalg.norm(h - np.eye(3)) > max_homography_deviation:
            return (AlignmentStatus.HOMOGRAPHY_FAILED, None, None, keypoints1, keypoints2, matches)

        # Use homography to warp img2 to align with img1
        aligned_img2 = cv2.warpPerspective(img2, h, (width, height))

        # Create a mask where the warping left empty spaces
        mask = np.zeros_like(aligned_img2[:, :, 0])
        mask[aligned_img2[:, :, 0] == 0] = 255

        return (AlignmentStatus.SUCCESS, aligned_img2, mask, keypoints1, keypoints2, matches)

    def align_images(
        self, ref_frame: Union[np.ndarray, str], frame_list: List[np.ndarray]
    ) -> List[Tuple[AlignmentStatus, Any, Any, Any, Any, Any]]:
        """
        Aligns the frames of frame_list to the ref_frame using ORB features.

        Args:
            ref_frame: The reference frame.
            frame_list: List of frames in cv2 image format.

        Returns:
            For each frame in frame list a Tuple with following entries:
                AlignmentStatus: The AlignmentStatus.
                aligned_frame: The aligned frame as a np.ndarray.
                mask: Mask where the alignment produced empty space as a np.ndarray.
                keypoints_ref: The keypoints in the reference image.
                keypoints_frame: The keypoints in the frame.
                matches: The comptued matches.
            If the AlignmentStatus of an entry is not AlignmentStatus.SUCCESS, the other entries in the
            Tuple are undefined.
        """
        min_matches = self.params.get("min_matches", 10)
        max_homography_deviation = self.params.get("max_homography_deviation", 10)
        check_keypoint_distance = self.params.get("check_keypoint_distance", True)
        max_keypoint_distance = self.params.get("max_keypoint_distance", 7)
        diasble_tqdm = self.params.get("diasble_tqdm", False)

        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create()
        # Match descriptors using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Reference frame
        img1 = process_input(ref_frame)
        height, width, _ = img1.shape
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)

        return_list: List[Tuple[AlignmentStatus, np.ndarray, np.ndarray, Any, Any, Any]] = []

        for img2 in tqdm(frame_list, desc="Matching frames to ref_frame", disable=diasble_tqdm):
            res = self._align_images(
                orb=orb,
                bf=bf,
                keypoints1=keypoints1,
                descriptors1=descriptors1,
                height=height,
                width=width,
                img2=img2,
                min_matches=min_matches,
                max_homography_deviation=max_homography_deviation,
                check_keypoint_distance=check_keypoint_distance,
                max_keypoint_distance=max_keypoint_distance,
            )
            return_list.append(res)

        return return_list

    def align_images_from_video_to_video(
        self, ref_frame: Union[np.ndarray, str], input_video: str, output_video: str, fps: int = 30
    ) -> List[Tuple[AlignmentStatus, Any, Any, Any]]:
        """
        Aligns the frames of frame_list to the ref_frame using ORB features.

        Args:
            ref_frame: The reference frame.
            input_video: Path to the input video.
            output_video: Path to the output video.
            fps: The frames per second.

        Returns:
            For each frame in frame list a Tuple with following entries:
                AlignmentStatus: The AlignmentStatus.
                keypoints_ref: The keypoints in the reference image.
                keypoints_frame: The keypoints in the frame.
                matches: The comptued matches.
            If the AlignmentStatus of an entry is not AlignmentStatus.SUCCESS, the other entries in the
            Tuple are undefined.
        """
        min_matches = self.params.get("min_matches", 10)
        max_homography_deviation = self.params.get("max_homography_deviation", 10)
        check_keypoint_distance = self.params.get("check_keypoint_distance", True)
        max_keypoint_distance = self.params.get("max_keypoint_distance", 7)
        diasble_tqdm = self.params.get("diasble_tqdm", False)
        save_failed_alignments = self.params.get("save_failed_alignments", False)

        return_list: List[Tuple[AlignmentStatus, Any, Any, Any]] = []
        with video_capture_context(input_video) as cap:
            # Detect ORB keypoints and descriptors
            orb = cv2.ORB_create()
            # Match descriptors using BFMatcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

            # Reference frame
            img1 = process_input(ref_frame)
            height, width, _ = img1.shape
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with video_writer_context(output_video, width=width, height=height, fps=fps) as video_writer:
                for _ in tqdm(range(length), desc="Matching frames to ref_frame", disable=diasble_tqdm):
                    succ, frame, _, kp1, kp2, matches = self._align_images(
                        orb=orb,
                        bf=bf,
                        keypoints1=keypoints1,
                        descriptors1=descriptors1,
                        height=height,
                        width=width,
                        img2=cap,
                        min_matches=min_matches,
                        max_homography_deviation=max_homography_deviation,
                        check_keypoint_distance=check_keypoint_distance,
                        max_keypoint_distance=max_keypoint_distance,
                    )
                    if succ == AlignmentStatus.SUCCESS or save_failed_alignments:
                        video_writer.write(frame)
                    # Dropping image and mask
                    res = (succ, kp1, kp2, matches)
                    return_list.append(res)

        return return_list
