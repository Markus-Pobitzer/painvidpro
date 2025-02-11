"""Base class for the processor."""

from typing import Any, Dict, List, Tuple


class ProcessorBase:
    def __init__(self):
        """Base class."""
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

    def process(self, video_dir_list: List[str], batch_size: int = -1) -> List[bool]:
        """
        Processes the videos stored under video_dir_list.

        Args:
            video_dir_list: List of paths where the videos are stored.
            batch_size: Batch size for internal computation, in case the processor supports it.

        Returns:
            A list of bools, indidcating for each element in video_dir_list, if the processing
            was successfull.
        """
        raise NotImplementedError("This method should be implemented by the child class.")
