import cv2
import numpy as np
from numpy.typing import NDArray
from typing import Literal
from .utils_image import DegradationOutput, uint2tensor

def jpeg_pipeline(
    H_img: NDArray[np.uint8],
    quality: int | None = None,
    quality_min: int = 5,
    quality_max: int = 95,
    subsampling: Literal["444", "422", "420", "411", "440"] = "420",
) -> DegradationOutput:
    _quality = quality if quality is not None else np.random.randint(quality_min, quality_max + 1)
    subsampling_map = {
        "444": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444,
        "422": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_422,
        "420": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420,
        "411": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_411,
        "440": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_440,
    }
    params = [
        cv2.IMWRITE_JPEG_QUALITY, _quality,
        cv2.IMWRITE_JPEG_SAMPLING_FACTOR, subsampling_map[subsampling]
    ]

    _, encimg = cv2.imencode(".jpg", cv2.cvtColor(H_img, cv2.COLOR_RGB2BGR), params)
    L_img = cv2.cvtColor(cv2.imdecode(encimg, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.uint8)

    H_img_tensor = uint2tensor(H_img)
    L_img_tensor = uint2tensor(L_img)

    return DegradationOutput(
        H_img=H_img_tensor,
        L_img=L_img_tensor,
        R_img=L_img_tensor,
        type=3,
    )


def main():
    from .utils_image import imread_uint_3, imshow, tensor2float
    from pathlib import Path

    image = imread_uint_3(Path("src/utils/test.png"))
    mosaic_return = jpeg_pipeline(image)
    imshow(
        [
            tensor2float(mosaic_return.H_img),
            tensor2float(mosaic_return.L_img),
            tensor2float(mosaic_return.R_img),
        ]
    )


if __name__ == "__main__":
    main()
