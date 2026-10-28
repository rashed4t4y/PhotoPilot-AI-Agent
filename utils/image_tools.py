"""Local JPEG/PNG validation and quality measurements; no API or disk writes.

Brightness is mean grayscale intensity (0–255); contrast is its population
standard deviation. Blur uses the population variance of the four-neighbour
Laplacian (up + down + left + right - 4 * centre), excluding the outer border.
Measurements use EXIF-oriented pixels at original resolution, converted to RGB;
transparency is composited on white. No resizing or colour-profile correction is
performed. Blur scores depend on resolution, texture, noise, and subject matter.
The thresholds below are illustrative heuristics, not calibrated quality gates
or promises of stock-platform acceptance.
"""

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import BinaryIO
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


MAX_FILE_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 40_000_000
SUPPORTED_FORMATS = frozenset({"JPEG", "PNG"})
ImageSource = bytes | bytearray | str | Path | BinaryIO


class ImageValidationError(ValueError):
    """An upload is unsupported, damaged, or exceeds the configured limits."""


@dataclass
class ValidatedImage:
    """Decoded, independent RGB image; the caller owns and may close it."""

    image: Image.Image
    file_format: str
    file_size_bytes: int


def _read_bytes(source: ImageSource, limit: int) -> bytes:
    if isinstance(source, (bytes, bytearray)):
        if len(source) > limit:
            raise ImageValidationError(f"Image exceeds the {limit}-byte file limit.")
        data = bytes(source)
    elif isinstance(source, (str, Path)):
        with open(source, "rb") as stream:
            data = stream.read(limit + 1)
    else:
        # Streamlit UploadedFile and BytesIO support seek/tell. Restore the
        # cursor even on failure so validation does not disrupt the preview.
        position = source.tell()
        try:
            source.seek(0)
            data = source.read(limit + 1)
        finally:
            source.seek(position)
    if not isinstance(data, bytes):
        raise ImageValidationError("Provide a binary image stream.")
    if not data:
        raise ImageValidationError("The image file is empty.")
    if len(data) > limit:
        raise ImageValidationError(f"Image exceeds the {limit}-byte file limit.")
    return data


def validate_image(
    source: ImageSource,
    *,
    max_file_bytes: int = MAX_FILE_BYTES,
    max_pixels: int = MAX_PIXELS,
) -> ValidatedImage:
    """Read without modifying the source; verify integrity and fully decode.

    Accept bytes, paths, or seekable binary streams. Only single-frame JPEG and
    PNG are supported. Pixel limits are checked before loading the raster.
    Pillow's own decompression-bomb safeguards remain enabled. Filesystem access
    errors remain OSError so callers can distinguish them from invalid uploads.
    """
    for name, value in (("max_file_bytes", max_file_bytes), ("max_pixels", max_pixels)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    data = _read_bytes(source, max_file_bytes)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as probe:
                file_format = probe.format
                if file_format not in SUPPORTED_FORMATS:
                    raise ImageValidationError("Only JPEG and PNG images are supported.")
                if probe.width * probe.height > max_pixels:
                    raise ImageValidationError(f"Image exceeds the {max_pixels}-pixel limit.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ImageValidationError("Animated images are not supported.")
                probe.verify()
            # verify() invalidates the decoder; reopen and load to also catch
            # truncated image data that a header/integrity check may miss.
            with Image.open(BytesIO(data)) as decoded:
                decoded.load()
                oriented = ImageOps.exif_transpose(decoded)
                try:
                    if "A" in oriented.getbands() or "transparency" in oriented.info:
                        with oriented.convert("RGBA") as rgba:
                            with Image.new("RGBA", rgba.size, "white") as background:
                                with Image.alpha_composite(background, rgba) as composite:
                                    rgb = composite.convert("RGB")
                    else:
                        rgb = oriented.convert("RGB")
                finally:
                    oriented.close()
    except ImageValidationError:
        raise
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ImageValidationError("Image exceeds Pillow's safe decoded-size limit.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise ImageValidationError("Image is damaged or cannot be decoded.") from exc
    return ValidatedImage(rgb, file_format, len(data))


def check_image_quality(source: ImageSource, **validation_limits: int) -> dict:
    """Return JSON/CSV-friendly measurements, method notes, and heuristic flags.

    Dimensions reflect EXIF orientation. Images smaller than 3 pixels on either
    axis have no interior Laplacian samples, so their blur score is None.
    Full-size float arrays are avoided by accumulating row-wise statistics.
    """
    validated = validate_image(source, **validation_limits)
    try:
        width, height = validated.image.size
        with validated.image.convert("L") as gray:
            pixels = np.asarray(gray)
        brightness = float(np.mean(pixels))
        contrast = float(np.std(pixels))
        blur_score = None
        if width >= 3 and height >= 3:
            total = squares = 0.0
            count = (width - 2) * (height - 2)
            for y in range(1, height - 1):
                row = pixels[y].astype(np.float64)
                laplacian = (
                    pixels[y - 1, 1:-1].astype(np.float64)
                    + pixels[y + 1, 1:-1]
                    + row[:-2] + row[2:] - 4 * row[1:-1]
                )
                total += float(laplacian.sum())
                squares += float(np.dot(laplacian, laplacian))
            blur_score = max(0.0, squares / count - (total / count) ** 2)
        flags = []
        if brightness < 40:
            flags.append("Possibly dark: mean grayscale brightness is below 40/255.")
        elif brightness > 215:
            flags.append("Possibly bright: mean grayscale brightness exceeds 215/255.")
        if contrast < 20:
            flags.append("Possibly low contrast: grayscale standard deviation is below 20.")
        if blur_score is not None and blur_score < 100:
            flags.append("Possible blur or smooth content: Laplacian variance is below 100.")
        return {
            "width": width,
            "height": height,
            "megapixels": width * height / 1_000_000,
            "file_format": validated.file_format,
            "file_size_bytes": validated.file_size_bytes,
            "brightness_mean": brightness,
            "contrast_stddev": contrast,
            "laplacian_variance": blur_score,
            "heuristic_flags": flags,
            "measurement_notes": (
                "EXIF-oriented, original-resolution grayscale; transparency on white. "
                "Brightness: mean 0–255; contrast: population standard deviation. "
                "Blur: population variance of four-neighbour Laplacian, border excluded; "
                "unavailable below 3x3 pixels. No colour-profile correction."
            ),
            "limitations": (
                "Flags are uncalibrated heuristics, not pass/fail judgments. Blur scores "
                "depend on resolution, texture and noise; smooth scenes can score low. "
                "These checks do not guarantee stock-platform acceptance."
            ),
        }
    finally:
        validated.image.close()
