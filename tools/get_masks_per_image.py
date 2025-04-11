# Given a folder path, show a summary of the different sizes of the images
# in the folder (width x height: number of images).
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from tqdm import tqdm


def get_number_of_masks(masks_file_path: Path):
    masks_npz = np.load(str(masks_file_path))
    masks_npz_keys = list(masks_npz.keys())
    masks = masks_npz[masks_npz_keys[0]]

    slices = masks.shape[2]
    progressbar = tqdm(desc='Looking for non-empty masks', total=slices, position=2, leave=False)

    number_of_masks = 0
    for index in range(slices):
        mask = masks[:, :, index]
        labels = np.unique(mask)
        if labels.size > 1:
            number_of_masks += 1

        progressbar.update()

    progressbar.close()

    result = {
        'path': str(masks_file_path),
        'slices': slices,
        'masks': number_of_masks
    }
    return result


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
        'working_data/covid/masks_coronacases_010.npz',
        'working_data/covid/masks_radiopaedia_4_85506_1.npz',
        'working_data/covid/masks_radiopaedia_7_85703_0.npz',
        'working_data/covid/masks_radiopaedia_10_85902_1.npz',
        'working_data/covid/masks_radiopaedia_10_85902_3.npz',
        'working_data/covid/masks_radiopaedia_14_85914_0.npz',
        'working_data/covid/masks_radiopaedia_27_86410_0.npz',
        'working_data/covid/masks_radiopaedia_29_86490_1.npz',
        'working_data/covid/masks_radiopaedia_29_86491_1.npz',
        'working_data/covid/masks_radiopaedia_36_86526_0.npz',
        'working_data/covid/masks_radiopaedia_40_86625_0.npz',
    ]

    progress_bar_description = 'Getting number of masks in masks files'
    items = len(masks_file_paths)
    progressbar = tqdm(desc=progress_bar_description, position=1, total=items, leave=True)

    number_of_masks_list = []
    for masks_file_path in masks_file_paths:
        progressbar.set_description(f'Getting number of masks in {masks_file_path}')
        number_of_masks = get_number_of_masks(Path(masks_file_path))
        number_of_masks_list.append(number_of_masks)

        progressbar.update()

    progressbar.set_description(progress_bar_description)
    progressbar.close()

    df = pd.DataFrame(number_of_masks_list)
    df.to_csv('comparison/number_of_masks.csv', index=False, float_format='%.0f')
    df.to_excel('comparison/number_of_masks.xlsx', index=False, float_format='%.0f')

    print(number_of_masks_list)


if __name__ == '__main__':
    main()
