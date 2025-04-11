"""
Process every single volume in a dataset and save the mask sizes in a CSV file
for later analysis. For this script, the mask size is the number of values
different from zero in the mask.
"""
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from data_model.dataset import Dataset

logger = logging.getLogger(__name__)



def get_masks_sizes(masks_file_path: Path) -> list:
    """
    Get the mask sizes for a given masks file.

    :param masks_file_path: path to the masks file to process.

    :return: list of dictionaries with the name of the processed file, the
    slice number and the size of the mask.
    """

    logger.info('Get masks sizes')
    logger.debug(f'masks_file_path('
                 f'masks_file_path="{masks_file_path}")')

    masks_npz = np.load(str(masks_file_path))
    masks_npz_keys = list(masks_npz.keys())
    masks = masks_npz[masks_npz_keys[0]]

    items = masks.shape[2]
    progress_bar = tqdm(desc='Getting mask sizes', total=items, leave=False, position=2)

    logger.info(f'Number of masks to process: {items}')

    mask_sizes = []
    for index in range(items):
        logger.info(f'Processing mask {index}')

        mask = masks[:, :, index]
        mask_size = np.count_nonzero(mask)
        mask_sizes.append({
            'Volume': masks_file_path.name,
            'Slice number': index,
            'Mask size': mask_size
        })

        progress_bar.update()

    progress_bar.close()

    return mask_sizes


def save_mask_sizes(dataset: Dataset):
    """
    Save the mask sizes in a CSV file for a given dataset.

    :param dataset: Dataset to process.
    """

    logger.info('Save mask sizes')
    logger.debug(f'save_mask_sizes('
                 f'dataset={dataset.description})')

    dataset_path = WorkingDataBasePath / dataset.dataset_path
    masks_files_paths = sorted(dataset_path.glob(dataset.masks_file_pattern))

    items = len(masks_files_paths)
    progress_bar = tqdm(desc='Processing masks files', total=items, leave=False, position=1)

    logger.info(f'Number of files to process: {items}')

    masks_sizes = []
    for masks_file_path in masks_files_paths:
        logger.info(f'Processing file "{masks_file_path}"')

        progress_bar.set_description(f'Processing {masks_file_path.name}')
        mask_sizes = get_masks_sizes(masks_file_path)
        masks_sizes = masks_sizes + mask_sizes
        progress_bar.update()

    progress_bar.set_description('Processing masks files')
    progress_bar.close()

    result_folder_path = OutputBasePath / dataset.code
    if not result_folder_path.exists():
        logger.info(f'Creating folder "{result_folder_path}"')
        result_folder_path.mkdir(parents=True)

    result_file_path = result_folder_path / Path(f'{dataset.code}_masks_sizes.csv')
    logger.info(f'Saving masks sizes to file "{result_file_path}"')

    masks_sizes_df = pd.DataFrame.from_records(masks_sizes)
    masks_sizes_df.to_csv(result_file_path, index=False)


def main():
    """
    Main function to process all the datasets and save the mask sizes in a CSV
    file. Adjust the dataset list with all the datasets you want to process.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename='debug.log',
            level=logging.DEBUG,
            format='%(asctime)-15s %(levelname)8s %(name)s %(message)s')

    logger.info('Start saving dataset masks sizes process')
    logger.debug('main()')

    datasets = [Dataset.Coronacases, Dataset.Radiopaedia, Dataset.Montgomery]

    items = len(datasets)
    progress_bar = tqdm(desc='Getting dataset masks sizes', total=items, leave=True, position=0)

    for dataset in datasets:
        progress_bar.set_description(f'Processing {dataset.description} dataset')
        save_mask_sizes(dataset)
        progress_bar.update()

    progress_bar.set_description('Getting dataset masks sizes')
    progress_bar.close()


if __name__ == '__main__':
    LoggingEnabled = False
    WorkingDataBasePath = Path('working_data/')
    OutputBasePath = Path('comparison/sam2_delta_report_image/')

    main()
