"""
Save the slices of the images as a series of JPG files, so that SAM 2 can
process them.

Split the resulting series in to parts, using the center slice as the seed.
This seed should be the first frame of each of the parts.

Optionally, insert a fake seed instead of using the center slice as the seed.
In both halves of the video, an extra frame is inserted before the seed.
"""
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from data_model.dataset import Dataset
from process_image import load_image, load_image_slice, load_masks, load_masks_slice
from tools.image_slice import ImageSlice

logger = logging.getLogger(__name__)

ImageKey = 'image'
MasksKey = 'masks'
FramesKey = 'frames'
SeedFrameKey = 'seed_frame'

FakeSeedKey = 'fake_seed'
FakeSeedImageKey = 'fake_seed_image'
FakeSeedMasksKey = 'fake_seed_masks'
FakeSeedSliceKey = 'fake_seed_slice'
FakeSeedFrameKey = 'fake_seed_frame'


def slice_to_jpg(
        points: np.array,
        labeled_points: np.array,
        apply_windowing: bool,
        slice_number: int,
        output_path: Path,
        items_digits: int
):
    """
    Save a single NPZ slice to a JPG files, so that SAM 2 can process it.

    :param points: slice to save as JPG.
    :param labeled_points: masks for given slice.
    :param apply_windowing: whether windowing should be applied or not.
    :param slice_number: number of the slice to save.
    :param output_path: path to the folder where the JPG files will be saved.
    :param items_digits: digits in the number of frames of the video, so zero
    padding can be used.
    """

    logger.info('Save NPZ as JPGs')
    logger.debug(f'save_mask_sizes('
                 f'points={len(points)}, '
                 f'labeled_points={len(labeled_points)}, '
                 f'apply_windowing={apply_windowing}, '
                 f'slice_number={slice_number}, '
                 f'output_path="{output_path}", '
                 f'items_digits={items_digits})')

    image_slice = ImageSlice(
        points=points,
        labeled_points=labeled_points,
        apply_windowing=apply_windowing,
        use_bounding_box=True,
        use_masks_contours=False)

    jpeg = Image.fromarray(image_slice.processed_points)
    jpeg_stem = str(slice_number).zfill(items_digits)
    jpeg.save(f'{output_path}/{jpeg_stem}.jpg')


def npz_to_jpgs(dataset: Dataset, npz_name: str) -> (Path, Path, Path):
    """
    Save the slices of a given NPZ file as a series of JPG files, so that SAM 2
    can process them.

    :param dataset: dataset to process.
    :param npz_name: name of the NPZ file to process.

    :return: path to the folder where the JPG files were saved.
    """

    logger.info('Save NPZ as JPGs')
    logger.debug(f'npz_to_jpgs('
                 f'dataset={dataset.description}, '
                 f'npz_name="{npz_name}")')

    image_file_path = WorkingDataBasePath / dataset.dataset_path / f'image_{npz_name}.npz'
    masks_file_path = WorkingDataBasePath / dataset.dataset_path / f'masks_{npz_name}.npz'

    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    image_file_stem = image_file_path.stem
    output_path = image_file_path.parent / OutputFolderPath / image_file_stem
    if not output_path.exists():
        output_path.mkdir(parents=True)

    items = image.shape[-1]
    items_digits = len(str(items))

    progress_bar = tqdm(desc='Saving slice as JPG', total=items, position=2, leave=False)

    for slice_number in range(items):
        points = load_image_slice(image=image, slice_number=slice_number)
        labeled_points = load_masks_slice(masks=masks, slice_number=slice_number)

        slice_to_jpg(
            points=points,
            labeled_points=labeled_points,
            apply_windowing=dataset.apply_windowing,
            slice_number=slice_number,
            output_path=output_path,
            items_digits=items_digits
        )

        progress_bar.update()

    progress_bar.close()

    return (
        image_file_path,
        masks_file_path,
        output_path
    )


def split_video(output_path: Path) -> (int, int):
    """
    Split the resulting series in to parts, using the center slice as the seed.
    This seed should be the first frame of each of the parts.

    :param output_path: path to the folder where the JPG frames to split in
    two halves are stored.
    """

    logger.info('Split video')
    logger.debug(f'split_video('
                 f'output_path="{output_path}")')

    frame_paths = sorted(output_path.glob('*.jpg'))
    frames = len(frame_paths)
    items_digits = len(str(frames))
    seed_frame = frames // 2

    half1_frames = frame_paths[:seed_frame + 1]
    half2_frames = frame_paths[seed_frame:]

    half1_path = output_path / Path('half1')
    if not half1_path.exists():
        half1_path.mkdir(parents=True)
    half2_path = output_path / Path('half2')
    if not half2_path.exists():
        half2_path.mkdir(parents=True)

    [shutil.copy(frame, half1_path) for frame in half1_frames]
    [shutil.copy(frame, half2_path) for frame in half2_frames]

    half1_frame_paths = sorted(half1_path.glob('*.jpg'))
    for frame_path in half1_frame_paths:
        temporary_frame_path = frame_path.parent / f'{frame_path.stem}_temporary.jpg'
        shutil.move(frame_path, temporary_frame_path)

    half1_frame_paths = sorted(half1_path.glob('*.jpg'))
    items = len(half1_frame_paths)
    for index, frame_path in enumerate(half1_frame_paths):
        final_frame_path = frame_path.parent / f'{str(items - 1 - index).zfill(items_digits)}.jpg'
        shutil.move(frame_path, final_frame_path)

    half2_frame_paths = sorted(half2_path.glob('*.jpg'))
    for index, frame_path in enumerate(half2_frame_paths):
        final_frame_path = frame_path.parent / f'{str(index).zfill(items_digits)}.jpg'
        shutil.move(frame_path, final_frame_path)

    # Move all frames to a separate folder
    all_path = output_path / Path('all')
    if not all_path.exists():
        all_path.mkdir(parents=True)

    for index, frame_path in enumerate(frame_paths):
        shutil.move(frame_path, all_path)

    return frames, seed_frame


def save_video_information(
        npz_name: str,
        output_path: Path,
        frames: int,
        seed_frame: int
):
    video_information = {
        ImageKey: f'working_data/covid/image_{npz_name}.npz',
        MasksKey: f'working_data/covid/masks_{npz_name}.npz',
        FramesKey: frames,
        SeedFrameKey: seed_frame
    }
    df = pd.DataFrame(video_information, index=[0])
    df.to_csv(output_path / 'video_information.csv', index=False)


def get_npz_names(dataset: Dataset) -> list:
    """
    Get the names of the NPZ files in a given dataset.

    :param dataset: dataset to process.

    :return: list of paths to the NPZ files in the given dataset.
    """

    logger.info('Get NPZ names')
    logger.debug(f'get_npz_names('
                 f'dataset={dataset.description})')

    dataset_path = WorkingDataBasePath / dataset.dataset_path
    npzs_paths = sorted(dataset_path.glob(dataset.image_file_pattern))
    npz_stems = [npz_path.stem for npz_path in npzs_paths]
    npz_names = [npz_stem.replace('image_', '') for npz_stem in npz_stems]

    return npz_names


# region Fake seed

def insert_fake_seed_in_video_half(fake_seed_path: Path, output_path: Path):
    """
    Insert the given fake seed in the video half, displacing all the existing
    frames.

    :param fake_seed_path: path to the fake seed to insert in the video.
    :param output_path: path where the video to modify is stored.
    """

    logger.info('Insert fake seed in video half')
    logger.debug(f'insert_fake_seed_in_video_half('
                 f'fake_seed_path="{fake_seed_path}", '
                 f'output_path="{output_path}")')

    items = len(list(output_path.glob('*.jpg')))
    items_digits = len(str(items))

    frame_paths = sorted(output_path.glob('*.jpg'), reverse=True)

    progress_bar = tqdm(desc=f'Inserting fake seed in {output_path.stem}', total=items, position=2, leave=False)

    for frame_path in frame_paths:
        stem = frame_path.stem
        frame_number = int(stem)
        stem_plus_one = str(frame_number + 1).zfill(items_digits)
        frame_path_plus_one = output_path / f'{stem_plus_one}.jpg'
        shutil.move(frame_path, frame_path_plus_one)

        progress_bar.update()

    progress_bar.close()

    shutil.copy(fake_seed_path, output_path / f'{'0'.zfill(items_digits)}.jpg')


def update_video_information(
        fake_seed_paths: dict,
        output_path: Path
):
    """
    Include the path to the fake seed in the video information CSV.

    :param fake_seed_paths: paths to the fake seed to insert in the video
    (the image, the masks, and the slice).
    :param output_path: path where the video to modify is stored.
    """

    logger.info('Update video information')
    logger.debug(f'update_video_information('
                 f'fake_seed_paths={fake_seed_paths}, '
                 f'output_path="{output_path}")')

    fake_seed_image = fake_seed_paths[FakeSeedImageKey]
    fake_seed_masks = fake_seed_paths[FakeSeedMasksKey]
    fake_seed_slice = fake_seed_paths[FakeSeedSliceKey]
    fake_seed_frame = fake_seed_paths[FakeSeedFrameKey]

    df = pd.read_csv(output_path / 'video_information.csv')
    df[FakeSeedKey] = True
    df[FakeSeedImageKey] = fake_seed_image
    df[FakeSeedMasksKey] = fake_seed_masks
    df[FakeSeedSliceKey] = fake_seed_slice
    df[FakeSeedFrameKey] = fake_seed_frame
    df.to_csv(output_path / 'video_information.csv', index=False)


def insert_fake_seed_in_video(
        fake_seed_paths: dict,
        output_path: Path):
    """
    Insert the given fake seed in the video. It will displace all the frames in
    the both halves. Include the path to the fake seed in the video information
    CSV.

    :param fake_seed_paths: paths to the fake seed to insert in the video
    (the image, the masks, and the slice).
    :param output_path: path where the video to modify is stored.
    """

    logger.info('Insert fake seed in video')
    logger.debug(f'insert_fake_seed_in_video('
                 f'fake_seed_paths={fake_seed_paths}, '
                 f'output_path="{output_path}")')

    fake_seed_slice_path = Path(fake_seed_paths[FakeSeedSliceKey])

    insert_fake_seed_in_video_half(fake_seed_slice_path, output_path / 'half1')
    insert_fake_seed_in_video_half(fake_seed_slice_path, output_path / 'half2')
    update_video_information(
        fake_seed_paths=fake_seed_paths,
        output_path=output_path
    )


# endregion

def process_npzs(dataset: Dataset, npz_names: list, fake_seed_paths: dict):
    """
    Process the NPZ files in a given dataset, saving them as a series of JPG
    files, and splitting the resulting series in to parts, using the center
    slice as the seed. This seed should be the first frame of each of the
    parts.

    :param dataset: dataset to process.
    :param npz_names: list of paths to the NPZ files in the given dataset.
    :param fake_seed_paths: paths to the fake seed to insert in the video
    (the image, the masks, and the slice).
    """

    logger.info('Process NPZs')
    logger.debug(f'process_npzs('
                 f'dataset={dataset.description}, '
                 f'npz_names={len(npz_names)} items, '
                 f'fake_seed_paths={fake_seed_paths})')

    items = len(npz_names)

    progress_bar = tqdm(desc='Processing NPZs', total=items, position=1, leave=False)

    for index, npz_name in enumerate(npz_names):
        progress_bar.set_description(f'Processing image {npz_name}')

        image_file_path, masks_file_path, output_path = npz_to_jpgs(dataset, npz_name)
        progress_bar.set_description(f'Splitting video {npz_name}')
        frames, seed_frame = split_video(output_path)
        save_video_information(npz_name, output_path, frames, seed_frame)

        if fake_seed_paths is not None:
            insert_fake_seed_in_video(
                fake_seed_paths=fake_seed_paths,
                output_path=output_path
            )

        progress_bar.update()

    progress_bar.set_description('Processing NPZs')
    progress_bar.close()


def main():
    """
    Main function to process all the datasets and save them as a series of JPG
    files. Adjust the dataset list with all the datasets you want to process.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename='debug.log',
            level=logging.DEBUG,
            format='%(asctime)-15s %(levelname)8s %(name)s %(message)s')

    logger.info('Start saving dataset as series of JPG files')
    logger.debug('main()')

    # List of datasets to process
    datasets = [
        Dataset.Coronacases,
        Dataset.Radiopaedia
    ]

    # List of fake seeds paths, one for dataset. Keep list empty for no fake seed.
    # fake_seeds_paths = []
    fake_seeds_paths = [
        {
            FakeSeedImageKey: f'{WorkingDataBasePath}/covid/image_coronacases_001.npz',
            FakeSeedMasksKey: f'{WorkingDataBasePath}/covid/masks_coronacases_001.npz',
            FakeSeedSliceKey: f'{WorkingDataBasePath}/covid/sam2_videos/image_coronacases_001/all/150.jpg',
            FakeSeedFrameKey: 150
        },
        {
            FakeSeedImageKey: f'{WorkingDataBasePath}/covid/image_radiopaedia_4_85506_1.npz',
            FakeSeedMasksKey: f'{WorkingDataBasePath}/covid/masks_radiopaedia_4_85506_1.npz',
            FakeSeedSliceKey: f'{WorkingDataBasePath}/covid/sam2_videos/image_radiopaedia_4_85506_1/all/19.jpg',
            FakeSeedFrameKey: 19
        }
    ]

    items = len(datasets)

    progress_bar = tqdm(desc='Processing datasets', total=items, position=0, leave=True)

    for index, dataset in enumerate(datasets):
        progress_bar.set_description(f'Processing dataset {dataset.description}')

        npz_names = get_npz_names(dataset)
        if len(fake_seeds_paths) == 0:
            fake_seed_paths = None
        else:
            fake_seed_paths = fake_seeds_paths[index]
        process_npzs(dataset, npz_names, fake_seed_paths)

        progress_bar.update()

    progress_bar.set_description(f'Processing datasets')
    progress_bar.close()


if __name__ == '__main__':
    LoggingEnabled = True
    WorkingDataBasePath = Path('working_data/')
    OutputFolderPath = Path('sam2_videos_fake_seed/')

    main()
