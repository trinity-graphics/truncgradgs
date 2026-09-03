# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "omegaconf>=2.3.0",
#     "opencv-python>=4.13.0.92",
#     "tqdm>=4.67.3",
#     "typer>=0.25.1",
# ]
# ///
import collections
import os
import shutil
from typing import Annotated, List, Optional

import cv2
import numpy as np
import typer
from omegaconf import OmegaConf
from rich.console import Console
from rich.pretty import Pretty
from tqdm import tqdm


console = Console()

app = typer.Typer()

def compute_motion_masks(
    frames: np.ndarray,
    pbar_desc: str,
    diff_thresh: float = 50,
    debug: bool = False,
    export_images: bool = False,
    export_path: Optional[str] = None,
) -> List[np.ndarray]:
    """
    Compute motion masks for a list of frames. The motion mask for frame i is computed
    as the absolute difference between frame i and frame i-1, followed by thresholding
    and morphological operations to get a binary mask.
    Args:
        frames: A numpy stack of frames for a camera.
    """
    h, w, _ = frames[0].shape
    masks = [np.zeros((h, w, 1), dtype=np.uint8)]  # First frame has no motion mask
    fifo = collections.deque(maxlen=5)  # For temporal median filtering of masks
    for i in tqdm(range(1, len(frames)), desc=pbar_desc, leave=False):
        diff = cv2.absdiff(
            cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY),
            cv2.cvtColor(frames[i - 1], cv2.COLOR_BGR2GRAY),
        )
        # Blur the difference image to reduce noise:
        diff = cv2.GaussianBlur(diff, (5, 5), 0)

        _, thresh = cv2.threshold(diff, diff_thresh, 255, cv2.THRESH_BINARY)
        # Dilate the thresholded image to fill in holes, then find contours and create a
        # mask:
        kernel = np.ones((7, 7), np.uint8)
        thresh_dilated = cv2.dilate(thresh, kernel, iterations=5)

        contours, _ = cv2.findContours(
            thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        mask = np.zeros((h, w, 1), dtype=np.uint8)
        for contour in contours:
            if cv2.contourArea(contour) > 10:  # Increased minimum contour area
                cv2.drawContours(
                    mask, [contour], -1, (255, 255, 255), thickness=cv2.FILLED
                )
        # cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
        fifo.append(mask)
        if len(fifo) > 1:
            mask = np.median(np.array(list(fifo)), axis=0).astype(np.uint8)
            mask = cv2.threshold(mask, diff_thresh, 255, cv2.THRESH_BINARY)[
                1
            ]  # Re-threshold after median
            kernel = np.ones((5, 5), np.uint8)
            thresh_dilated = cv2.dilate(mask, kernel, iterations=3)
            contours, _ = cv2.findContours(
                thresh_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            mask = np.zeros((h, w, 1), dtype=np.uint8)
            for contour in contours:
                if cv2.contourArea(contour) > 10:  # Increased minimum contour area
                    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

        if debug:
            # Visualize
            cv2.imshow("Frame", cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR))
            cv2.imshow("Diff", diff)
            cv2.imshow("Thresh", thresh)
            cv2.imshow("Motion Mask", mask)
            exit_ = -1
            while exit_ != ord("q"):
                exit_ = cv2.waitKey(1)
            cv2.destroyAllWindows()

        if export_images:
            if export_path is None:
                raise ValueError(
                    "export_path must be provided if export_images is True"
                )
            os.makedirs(export_path, exist_ok=True)
            cv2.imwrite(
                os.path.join(
                    export_path,
                    f"motion_mask_{i:04d}.png",
                ),
                mask,
            )
            cv2.imwrite(
                os.path.join(
                    export_path,
                    f"frame_{i:04d}.png",
                ),
                cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR),
            )

        masks.append(mask)
    return masks


@app.command()
def export_t_t(
    root: Annotated[str, typer.Argument(help="Path to the dataset root directory")],
    clean: Annotated[
        bool,
        typer.Option(help="Whether to clean existing motion masks before exporting"),
    ] = False,
    export_images: Annotated[
        bool,
        typer.Option(
            help="Whether to export motion masks as images (for debugging purposes)"
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            help="Whether to visualize the motion mask computation process (for debugging purposes)"
        ),
    ] = False,
):
    """
    Export motion masks from a dataset (Tanks & Temples).
    """
    raise NotImplementedError("Motion mask export for Tanks & Temples is not implemented yet.")

@app.command()
def export_ours(
    root: Annotated[str, typer.Argument(help="Path to the dataset root directory")],
    clean: Annotated[
        bool,
        typer.Option(help="Whether to clean existing motion masks before exporting"),
    ] = False,
    export_images: Annotated[
        bool,
        typer.Option(
            help="Whether to export motion masks as images (for debugging purposes)"
        ),
    ] = False,
    debug: Annotated[
        bool,
        typer.Option(
            help="Whether to visualize the motion mask computation process (for debugging purposes)"
        ),
    ] = False,
):
    """
    Export motion masks from a dataset (our synthetic benchmark).
    """
    from camera_conf import CameraConf
    from utils import get_camera_frames
    with open(os.path.join(root, "cameras_config.yaml"), "r") as f:
        schema = OmegaConf.structured(CameraConf)
        conf = OmegaConf.load(f)
        OmegaConf.merge(schema, conf)
        console.print(Pretty(conf))

    pbar = tqdm(
        total=conf.n_arrays * conf.n_cameras_per_array, desc="Processing cameras"
    )
    if os.path.exists(os.path.join(root, "motion_masks")) and clean:
        shutil.rmtree(os.path.join(root, "motion_masks"))
    os.makedirs(os.path.join(root, "motion_masks"), exist_ok=True)
    for array_id in range(conf.n_arrays):
        for camera_id in range(conf.n_cameras_per_array):
            masks_file_path = os.path.join(
                root,
                "motion_masks",
                f"array_{array_id:02d}_camera_{camera_id:03d}",
            )
            if os.path.isdir(masks_file_path) and not clean:
                pbar.update(1)
                continue
            elif not os.path.isdir(masks_file_path):
                os.makedirs(masks_file_path, exist_ok=True)
            img_frames = get_camera_frames(root, array_id, camera_id, leave_pbar=False)
            masks = compute_motion_masks(
                img_frames,
                f"Computing motion masks for array {array_id} camera {camera_id}",
                debug=debug,
                export_images=export_images and array_id == 0 and camera_id == 0,
                export_path=os.path.join(
                    root, "motion_masks", f"array_{array_id:02d}_camera_{camera_id}"
                ),
            )
            pbar.set_description_str(
                f"Exporting motion masks for array {array_id} camera {camera_id}..."
            )
            # np.savez_compressed(
            #     masks_file_path,
            #     masks=np.array(masks),
            # )
            for i, mask in enumerate(masks):
                np.savez_compressed(
                    os.path.join(masks_file_path, f"mask_{i:05d}.npz"), mask
                )
            pbar.update(1)
            pbar.set_description("Processing cameras")


if __name__ == "__main__":
    app()
