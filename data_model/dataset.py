import logging
from collections import namedtuple
from enum import Enum
from pathlib import Path

log = logging.getLogger(__name__)

DatasetItem = namedtuple(
    typename='DatasetItem',
    field_names=[
        'code',
        'description',
        'dataset_path',
        'image_file_pattern',
        'masks_file_pattern',
        'sam1_data_path',
        'sam2_data_path',
        'apply_windowing']
)


class Dataset(DatasetItem, Enum):
    """
    Possible datasets used to evaluate the segmentation process.
    """

    Coronacases = DatasetItem(
        code='coronacases',
        description='Coronacases',
        dataset_path=Path('covid'),
        image_file_pattern='image_coronacases_*.npz',
        masks_file_pattern='masks_coronacases_*.npz',
        sam1_data_path=Path('covid/results/sam1/2-coronacases/'),
        sam2_data_path=Path('covid/results/sam2/2-coronacases/'),
        apply_windowing=True
    )
    Radiopaedia = DatasetItem(
        code='radiopaedia',
        description='Radiopaedia',
        dataset_path=Path('covid'),
        image_file_pattern='image_radiopaedia_*.npz',
        masks_file_pattern='masks_radiopaedia_*.npz',
        sam1_data_path=Path('covid/results/sam1/3-radiopaedia/'),
        sam2_data_path=Path('covid/results/sam2/3-radiopaedia/'),
        apply_windowing=False
    )
    Montgomery = DatasetItem(
        code='montgomery',
        description='Montgomery',
        dataset_path=Path('montgomery'),
        image_file_pattern='image_*.npz',
        masks_file_pattern='masks_*.npz',
        sam1_data_path=Path('montgomery/results/sam1_all/'),
        sam2_data_path=Path('montgomery/results/sam2_all/'),
        apply_windowing=False
    )
    NotImplemented = DatasetItem(
        code='',
        description='',
        dataset_path=None,
        image_file_pattern='',
        masks_file_pattern='',
        sam1_data_path=None,
        sam2_data_path=None,
        apply_windowing=None
    )

    @classmethod
    def from_string(
            cls,
            code: str
    ):
        """
        Get an instance of this class from its code as a string.

        :param code: value of the code as a string.

        :return: instance of this enumeration.
        """

        log.info('Create instance of Dataset from its code')
        log.debug(f'Dataset.from_string('
                  f'code="{code}")')

        for dataset in list(Dataset):
            if code.lower() == dataset.code.lower():
                return dataset

        return cls.NotImplemented
