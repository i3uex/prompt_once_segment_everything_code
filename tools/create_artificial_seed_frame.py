# Create an artificial seed frame from the masks of the Coronacases dataset
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from tqdm import tqdm


def get_seed_mask(masks_file_path: Path):
    masks_npz = np.load(str(masks_file_path))
    masks_npz_keys = list(masks_npz.keys())
    masks = masks_npz[masks_npz_keys[0]]

    slices = masks.shape[2]
    seed_frame = slices // 2
    seed_mask = masks[:, :, seed_frame]
    seed_mask = seed_mask.astype(np.uint8)

    return seed_mask


def main():
    masks_file_paths = [
        'working_data/covid/masks_coronacases_001.npz',
        'working_data/covid/masks_coronacases_002.npz',
        'working_data/covid/masks_coronacases_003.npz',
        'working_data/covid/masks_coronacases_004.npz',
        'working_data/covid/masks_coronacases_005.npz',
        'working_data/covid/masks_coronacases_006.npz',
        'working_data/covid/masks_coronacases_007.npz',
        'working_data/covid/masks_coronacases_008.npz',
        'working_data/covid/masks_coronacases_009.npz',
        'working_data/covid/masks_coronacases_010.npz'
    ]
    output_path = Path('working_data/covid/artificial_seed_frames')

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    progress_bar_description = 'Getting number of masks in masks files'
    items = len(masks_file_paths)
    progressbar = tqdm(desc=progress_bar_description, position=1, total=items, leave=True)

    artificial_seed_frame = np.zeros((512, 512), dtype=np.uint8)
    for masks_file_path in masks_file_paths:
        progressbar.set_description(f'Getting seed masks in {masks_file_path}')
        seed_mask = get_seed_mask(Path(masks_file_path))

        artificial_seed_frame += seed_mask

        image = Image.fromarray(seed_mask)
        image_stem = Path(masks_file_path).stem
        image.save(f'{output_path}/{image_stem}_mask.jpg')

        progressbar.update()

    progressbar.set_description(progress_bar_description)
    progressbar.close()

    # image = exposure.adjust_gamma(artificial_seed_frame, gamma=2)
    # image = exposure.adjust_log(artificial_seed_frame, 1)
    pixels = ((artificial_seed_frame - artificial_seed_frame.min()) / (
                artificial_seed_frame.max() - artificial_seed_frame.min())) * 255
    pixels = pixels.astype(np.uint8)
    image = Image.fromarray(pixels)
    image.save(f'{output_path}/artificial_seed_frame_01.jpg')

    thresh = threshold_otsu(artificial_seed_frame)
    binary = artificial_seed_frame > thresh
    image = Image.fromarray(binary)
    image.save(f'{output_path}/artificial_seed_frame_02.jpg')


if __name__ == '__main__':
    main()
