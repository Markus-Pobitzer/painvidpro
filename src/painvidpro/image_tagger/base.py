"""Base class for the Keyframe detection."""

from typing import Any, Dict, Tuple

from PIL import Image


class ImageTaggerBase:
    def __init__(self):
        """Base class to detect keyframes."""
        self.params: Dict[str, Any] = {}

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
        raise NotImplementedError("This method should be implemented by the child class.")

    def predict(
        self,
        image: Image.Image,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Predict image description and tags.

        Args:
            image: PIL Image.

        Returns:
            Dict containting `image_description` key. Other keys dependant on child class.
        """
        raise NotImplementedError("This method should be implemented by the child class.")
