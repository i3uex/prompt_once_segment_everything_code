import logging
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


class Debug:
    enabled: bool
    image_file_path: Path
    masks_file_path: Path
    slice_number: int
    base_folder_path = Path('debug')
    folder_path: Path
    base_file_path: Path

    def __init__(
            self,
            enabled: bool,
            image_file_path: Path,
            masks_file_path: Path
    ):
        """
        Init Debug class instance.

        :param enabled: True if debug information should be created, False
        otherwise.
        :param image_file_path: path to the images file.
        :param masks_file_path: path to the masks file.
        """

        logger.info('Init Debug')
        logger.debug(f'Debug.__init__('
                     f'enabled={enabled}, '
                     f'image_file_path="{image_file_path}", '
                     f'mask_file_path="{masks_file_path}")')

        self.enabled = enabled
        self.image_file_path = image_file_path
        self.masks_file_path = masks_file_path

        if self.enabled:
            self.set_folder_path()

    def set_folder_path(self):
        self.folder_path = \
            self.image_file_path.parent / \
            self.base_folder_path / \
            Path(self.image_file_path.stem)
        self.folder_path.mkdir(parents=True, exist_ok=True)

    def set_slice_number(self, slice_number: int):
        self.slice_number = slice_number
        self.set_base_file_path(self.slice_number)

    def set_base_file_path(self, slice_number: int):
        base_file_name = f'slice_{slice_number}'
        self.base_file_path = self.folder_path / Path(base_file_name)

    def get_file_path(self, name_suffix: str, extension: str) -> Path:
        output_file_stem = f'{self.base_file_path.stem}_{name_suffix}'
        output_file_path = self.base_file_path \
            .with_stem(output_file_stem) \
            .with_suffix(extension)

        return output_file_path

    @staticmethod
    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
