import itertools
from typing import List, Sequence

import cv2
import numpy as np
import numpy.typing as npt

from ._base import BaseEncoder

__all__ = ["GaborEncoder", "OccludableGaborEncoder"]


def get_filters(
    phase_offsets: Sequence,
    orientations: Sequence,
    wavelengths: Sequence,
    kernel_size: int,
    spatial_wavelength: float = 1.0,
    aspect_ratio: float = 0.5,
    is_normalised: bool = True,
) -> List[np.ndarray]:
    """Gets Gabor filters for image transforms.

    Args:
        phase_offsets (Sequence): Phase offset of sinusoid in radians (psi).
        orientations (Sequence): Filter orientation in radians (theta).
        wavelengths (Sequence): Sinusoid wavelengths in px / cycle (lambd).
        kernel_size (int): Kernel size (ksize).
        spatial_wavelength (float, optional): Spatial bandwidth in octaves (b). Defaults to 1.0.
        aspect_ratio (float, optional): Filter aspect ratio (gamma). Defaults to 0.5.
        is_normalised (bool, optional): Normalises kernel.

    Returns:
        List[ndarray]: List of Gabor filters.
    """

    def create_filter(phase_offset, orientation, wavelength):
        sigma = (
            wavelength
            / np.pi
            * np.sqrt(np.log(2) / 2)
            * (2**spatial_wavelength + 1)
            / (2**spatial_wavelength - 1)
        )
        filter_kernel = cv2.getGaborKernel(
            (kernel_size, kernel_size),
            sigma,
            orientation,
            wavelength,
            aspect_ratio,
            phase_offset,
            ktype=cv2.CV_32F,
        )
        filter_kernel -= np.mean(filter_kernel)  # type: ignore
        if is_normalised:
            filter_kernel /= 1.0 * np.sqrt(np.sum(filter_kernel**2))
        return filter_kernel

    filters = [
        create_filter(phase_offset, orientation, wavelength)
        for phase_offset, orientation, wavelength in itertools.product(
            phase_offsets, orientations, wavelengths
        )
    ]
    return filters


class GaborEncoder(BaseEncoder):
    """Encoder to transform an input image into a driving stimulus using a set of Gabor filters.

    Attributes:
        filters (List[np.ndarray]): List of filters to be applied to the image.
        scale_factor (float): Scaling factor for activations.
        renormalise (bool, optional): Whether to renormalise the filtered images. Defaults to False.
        min_value (float, optional): Minimum value for the filtered images. Defaults to 0.
    """

    def __init__(
        self,
        phase_offsets: Sequence,
        orientations: Sequence,
        wavelengths: Sequence,
        kernel_size: int,
        scale_factor: float,
        renormalise: bool = False,
        min_value: float = 0,
        **filter_kwargs,
    ) -> None:
        self.filters = get_filters(
            phase_offsets, orientations, wavelengths, kernel_size, **filter_kwargs
        )
        self.scale_factor = scale_factor
        self.renormalise = renormalise
        self.min_value = min_value

    def transform(self, data: np.ndarray) -> npt.NDArray[np.float_]:
        filter_outputs = np.zeros([len(self.filters), *data.shape])
        stdev = np.std(data)
        if stdev > 0:
            image_ = data / stdev
            for idx, filter in enumerate(self.filters):
                filter_output = cv2.filter2D(image_, cv2.CV_64F, filter)  # type: ignore[attr-defined]
                filter_outputs[idx, ...] = filter_output
            filter_outputs[filter_outputs < self.min_value] = 0

        if self.renormalise:
            filter_outputs = np.array(
                [
                    filter_output / np.linalg.norm(filter_output)
                    if np.any(filter_output)
                    else filter_output
                    for filter_output in filter_outputs
                ]
            )

        stdev = np.std(filter_outputs)
        if stdev > 0:
            filter_outputs = filter_outputs / np.std(filter_outputs)
        return self.scale_factor * filter_outputs.flatten()


class OccludableGaborEncoder(GaborEncoder):
    """Extension of GaborEncoder that allows for post-process masking
    of the rate map to simulate occlusion.
    """

    def __init__(
        self,
        phase_offsets: Sequence,
        orientations: Sequence,
        wavelengths: Sequence,
        kernel_size: int,
        scale_factor: float,
        renormalise: bool = False,
        min_value: float = 0,
        mask_width: int = 0,
        **filter_kwargs,
    ) -> None:
        super().__init__(
            phase_offsets,
            orientations,
            wavelengths,
            kernel_size,
            scale_factor,
            renormalise=renormalise,
            min_value=min_value,
            **filter_kwargs,
        )
        self._mask_x_end: int = mask_width

    def set_occlusion_curtain(self, width_pixels: int) -> None:
        """Sets the width of the occlusion curtain starting from x=0."""
        self._mask_x_end = width_pixels

    def transform(self, data: np.ndarray) -> npt.NDArray[np.float_]:
        flat_output = super().transform(data)

        if self._mask_x_end <= 0:
            return flat_output

        # Reshape to restore spatial awareness
        # Shape is (N_filters, Height, Width)
        n_filters = len(self.filters)
        h, w = data.shape

        # Safety check to ensure reshape is valid
        if flat_output.size != (n_filters * h * w):
            raise ValueError(
                f"Output size {flat_output.size} does not match expected dims ({n_filters}, {h}, {w})"
            )

        shaped_output = flat_output.reshape((n_filters, h, w))

        # Apply the "Sliding Curtain" Mask
        # We zero out everything from x=0 to x=mask_end across all filters/rows
        shaped_output[:, :, : self._mask_x_end] = 0.0

        return shaped_output.flatten()
