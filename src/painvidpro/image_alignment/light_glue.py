"""Class for the image alignment."""

from typing import Any, Dict, List, Tuple, Union

import cv2
import numpy as np
from lightglue import LightGlue, SuperPoint
from lightglue.utils import numpy_image_to_torch, rbd
from tqdm import tqdm

from painvidpro.image_alignment.base import AlignmentStatus, ImageAlignmentBase
from painvidpro.utils.image_processing import process_input
from painvidpro.video_processing.utils import video_capture_context, video_writer_context


class ImageAlignmentLightGlue(ImageAlignmentBase):
    def __init__(self):
        """Class to align images."""
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
            "max_num_keypoints": 1024,
            "disable_tqdm": False,
            "save_failed_alignments": False,  # Saves the frames that failed to align on disk as is
        }

    def _check_keypoint_distance(
        self,
        keypoints0: np.ndarray,
        keypoints1: np.ndarray,
        matches: List[cv2.DMatch],
        max_keypoint_distance: float,
        min_matches: int,
    ) -> Tuple[bool, np.ndarray, np.ndarray, List[cv2.DMatch]]:
        """
        Extracts location of good matches and filters by distance.

        Args:
            keypoints0: Keypoints from the first image.
            keypoints1: Keypoints from the second image.
            matches: Matches between the keypoints.
            max_keypoint_distance: Maximum allowed distance between matched keypoints.
            min_matches: Minimum number of good matches required.

        Returns:
            - A boolean indicating if there are enough good matches after filtering.
            - List of points from the first image.
            - List of points from the second image.
            - List of good matches.
        """
        points0 = []
        points1 = []
        good_matches = []

        for match in matches:
            pt0 = keypoints0[match.queryIdx]
            pt1 = keypoints1[match.trainIdx]
            distance = np.linalg.norm(np.array(pt0) - np.array(pt1))
            if distance <= max_keypoint_distance:
                points0.append(pt0)
                points1.append(pt1)
                good_matches.append(match)

        ret_points0 = np.array(points1, dtype=np.float32)
        ret_points1 = np.array(points1, dtype=np.float32)

        # Check if there are enough good matches after filtering
        if len(good_matches) < min_matches:
            return False, ret_points0, ret_points1, good_matches

        return True, ret_points0, ret_points1, good_matches

    def _align_images(
        self,
        extractor: SuperPoint,
        matcher: LightGlue,
        feats0: Dict[str, Any],
        height: int,
        width: int,
        img2: Union[np.ndarray, str, cv2.VideoCapture],
        min_matches: int = 10,
        max_homography_deviation: int = 10,
        check_keypoint_distance: bool = True,
        max_keypoint_distance: float = 7.0,
    ):
        """
        Aligns the second image to the first image using keypoints and descriptors.

        Args:
            extractor: Keypoint detection and descriptor computation.
            matcher: LightGlue matcher for matching descriptors.
            feats0: Keypoints and descriptors from the first image.
            height: Height of the images.
            width: Width of the images.
            img2: Image to be aligned.
            min_matches: Minimum number of matches required.
            max_homography_deviation: Maximum allowed deviation for homography.
            check_keypoint_distance: Whether to check the distance between keypoints.
            max_keypoint_distance: Maximum allowed distance between matched keypoints.

        Returns:
            Tuple: Alignment status, aligned image, mask, keypoints from the first image, keypoints from the second image, and matches.
        """
        img2 = process_input(img2)
        image1 = numpy_image_to_torch(img2[..., ::-1]).cuda()

        feats1 = extractor.extract(image1)

        # Match the features
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]
        matches = matches01["matches"]

        # Extract location of good matches
        points0 = feats0["keypoints"][matches[..., 0]].cpu().numpy()
        points1 = feats1["keypoints"][matches[..., 1]].cpu().numpy()

        # Convert keypoints to cv2.KeyPoint objects
        ret_keypoints0 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in points0]
        ret_keypoints1 = [cv2.KeyPoint(x=pt[0], y=pt[1], size=1) for pt in points1]
        ret_matches = [cv2.DMatch(i, i, 0) for i in range(len(matches))]

        # Check if there are enough matches
        if len(matches) < min_matches:
            return (AlignmentStatus.NOT_ENOUGH_MATCHES, None, None, ret_keypoints0, ret_keypoints1, ret_matches)

        if check_keypoint_distance:
            succ, points0, points1, ret_matches = self._check_keypoint_distance(
                points0, points1, ret_matches, max_keypoint_distance, min_matches
            )
            if not succ:
                return (
                    AlignmentStatus.NOT_ENOUGH_GOOD_MATCHES,
                    None,
                    None,
                    ret_keypoints0,
                    ret_keypoints1,
                    ret_matches,
                )

        # Find homography matrix
        h, _mask = cv2.findHomography(points1, points0, cv2.RANSAC)

        # Check if the homography matrix represents a small alignment
        if np.linalg.norm(h - np.eye(3)) > max_homography_deviation:
            return (AlignmentStatus.HOMOGRAPHY_FAILED, None, None, ret_keypoints0, ret_keypoints1, ret_matches)

        # Use homography to warp img2 to align with img1
        aligned_img2 = cv2.warpPerspective(img2, h, (width, height))

        # Create a mask where the warping left empty spaces
        mask = np.zeros_like(aligned_img2[:, :, 0])
        mask[aligned_img2[:, :, 0] == 0] = 255

        return (AlignmentStatus.SUCCESS, aligned_img2, mask, ret_keypoints0, ret_keypoints1, ret_matches)

    def align_images(
        self, ref_frame: Union[np.ndarray, str], frame_list: Union[List[np.ndarray], List[str]]
    ) -> List[Tuple[AlignmentStatus, Any, Any, Any, Any, Any]]:
        """
        Aligns the frames of frame_list to the ref_frame using LightGlue.

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
        disable_tqdm = self.params.get("disable_tqdm", False)
        max_num_keypoints = self.params.get("max_num_keypoints", 1024)

        # Initialize the extractor and matcher
        extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()
        matcher = LightGlue(features="superpoint").eval().cuda()

        # Reference frame
        img1 = process_input(ref_frame)
        height, width, _ = img1.shape
        img1 = numpy_image_to_torch(img1[..., ::-1]).cuda()

        # Extract local features
        feats0 = extractor.extract(img1)

        return_list: List[Tuple[AlignmentStatus, np.ndarray, np.ndarray, Any, Any, Any]] = []

        for img2 in tqdm(frame_list, desc="Matching frames to ref_frame", disable=disable_tqdm):
            res = self._align_images(
                extractor=extractor,
                matcher=matcher,
                feats0=feats0,
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
        Aligns the frames of frame_list to the ref_frame using LightGlue.

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
        disable_tqdm = self.params.get("disable_tqdm", False)
        save_failed_alignments = self.params.get("save_failed_alignments", False)
        max_num_keypoints = self.params.get("max_num_keypoints", 1024)

        return_list: List[Tuple[AlignmentStatus, Any, Any, Any]] = []
        with video_capture_context(input_video) as cap:
            # Initialize the extractor and matcher
            extractor = SuperPoint(max_num_keypoints=max_num_keypoints).eval().cuda()
            matcher = LightGlue(features="superpoint").eval().cuda()

            # Reference frame
            img1 = process_input(ref_frame)
            height, width, _ = img1.shape
            img1 = numpy_image_to_torch(img1[..., ::-1]).cuda()
            # Extract local features
            feats0 = extractor.extract(img1)

            length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            with video_writer_context(output_video, width=width, height=height, fps=fps) as video_writer:
                for _ in tqdm(range(length), desc="Matching frames to ref_frame", disable=disable_tqdm):
                    succ, frame, _, kp1, kp2, matches = self._align_images(
                        extractor=extractor,
                        matcher=matcher,
                        feats0=feats0,
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
