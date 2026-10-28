"""Preliminary resolution assessment, separate from quality and acceptance."""
import math

from utils.image_tools import validate_image

DISCLAIMER = (
    "Automated resolution and quality checks do not guarantee platform acceptance. "
    "Licensing, intellectual property, content policies, file requirements and image quality "
    "still need review. This is not an official Shutterstock submission report."
)


def assess_resolution(width, height, minimum_mp=4.0, platform="Shutterstock") -> dict:
    """Compare verified original dimensions, without rounding or upscaling."""
    result = {
        "platform": platform, "width": width, "height": height, "megapixels": None,
        "minimum_megapixels": minimum_mp, "status": "failed",
        "resolution_result": "unavailable", "message": "", "disclaimer": DISCLAIMER,
    }
    if (type(minimum_mp) not in (int, float) or not math.isfinite(minimum_mp)
            or minimum_mp <= 0):
        result["message"] = "Resolution assessment unavailable: configure a finite positive minimum megapixel value."
        return result
    if any(type(value) is not int or value <= 0 for value in (width, height)):
        result["message"] = "Resolution assessment unavailable: valid original image dimensions could not be measured."
        return result
    result["megapixels"] = width * height / 1_000_000
    if result["megapixels"] < minimum_mp:
        result.update(
            status="needs_review", resolution_result="below_minimum",
            message=f"Below the configured {platform} minimum resolution. This image is not ready for submission at its current dimensions.",
        )
    else:
        result.update(status="completed", resolution_result="passed",
                      message="Resolution check passed. No other submission requirements have been established.")
    return result


def assess_image_readiness(source, minimum_mp=4.0, platform="Shutterstock") -> dict:
    """Measure an actual image; missing/corrupt/unsupported data never passes."""
    try:
        validated = validate_image(source)
        try:
            return assess_resolution(*validated.image.size, minimum_mp, platform)
        finally:
            validated.image.close()
    except (ValueError, OSError, TypeError, AttributeError):
        result = assess_resolution(None, None, minimum_mp, platform)
        result["message"] = "Resolution assessment unavailable: image is missing, invalid, unsupported or exceeds validation limits."
        return result
