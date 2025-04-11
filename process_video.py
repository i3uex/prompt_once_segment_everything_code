import logging
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from sam2.build_sam import build_sam2_video_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor
from tqdm import tqdm

from csv_keys import *
from data_model.dataset import Dataset
from data_processor.npz_to_jpgs import FakeSeedKey, FakeSeedFrameKey, SeedFrameKey, MasksKey, FakeSeedImageKey, \
    FakeSeedMasksKey
from process_image import load_image, load_masks, load_image_slice, load_masks_slice, \
    compare_original_and_predicted_masks
from sam_model import SamModel
from tools.debug import Debug
from tools.image_slice import ImageSlice
from tools.timestamp import Timestamp

logger = logging.getLogger(__name__)

LoggingEnabled = True

ShowPlots = False


def get_sam2_video_predictor() -> SAM2VideoPredictor:
    torch.autocast(device_type='cuda', dtype=torch.bfloat16).__enter__()
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    sam_model = SamModel.Large
    sam2_checkpoint = sam_model.checkpoint
    model_cfg = sam_model.configuration

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

    return video_predictor


def get_video_information(video_folder: Path) -> dict:
    df = pd.read_csv(video_folder / 'video_information.csv', index_col=False)
    list_of_dictionaries = df.to_dict('records')
    dictionary = list_of_dictionaries[0]

    return dictionary

def get_fake_seed(video_information: dict) -> bool:
    if FakeSeedKey in video_information.keys():
        fake_seed = video_information[FakeSeedKey]
    else:
        fake_seed = False
    return fake_seed

def get_frame_names(video_folder: Path) -> list:
    frame_names = [
        p for p in os.listdir(video_folder)
        if os.path.splitext(p)[-1] in ['.jpg', '.jpeg', '.JPG', '.JPEG']
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    return frame_names


def show_first_video_frame(video_folder: Path, video_information: dict, frame_names: list):
    video_name = video_folder.parent.stem
    seed_frame_index = video_information[SeedFrameKey]
    half_number = video_folder.stem[-1]
    frame_idx = 0
    plt.figure(figsize=(9, 6))
    plt.title(f'video: {video_name} - seed frame: {seed_frame_index} half: {half_number} - frame {frame_idx}')
    plt.imshow(Image.open(os.path.join(video_folder, frame_names[frame_idx])))
    plt.tight_layout()
    plt.show()


def get_image_slice(dataset: Dataset, video_information: dict) -> ImageSlice:
    fake_seed = get_fake_seed(video_information)

    if not fake_seed:
        image_file = video_information[ImageKey]
        masks_file = video_information[MasksKey]
        slice_number = video_information[SeedFrameKey]
    else:
        image_file = video_information[FakeSeedImageKey]
        masks_file = video_information[FakeSeedMasksKey]
        slice_number = video_information[FakeSeedFrameKey]

    image_file_path = Path(image_file)
    masks_file_path = Path(masks_file)

    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    image_slice_points = load_image_slice(image=image, slice_number=slice_number)
    image_slice_labeled_points = load_masks_slice(masks=masks, slice_number=slice_number)

    if dataset is Dataset.Coronacases:
        apply_windowing = True
    elif dataset is Dataset.Radiopaedia:
        apply_windowing = False
    else:
        raise ValueError(f'Unknown dataset: {dataset}')

    return ImageSlice(
        points=image_slice_points,
        labeled_points=image_slice_labeled_points,
        apply_windowing=apply_windowing,
        use_bounding_box=True,
        use_masks_contours=False)


def add_prompt_to_seed_frame(
        dataset: Dataset,
        video_folder: Path,
        video_information: dict,
        video_predictor: SAM2VideoPredictor,
        inference_state: dict,
        frame_names: list
):
    image_slice = get_image_slice(dataset=dataset, video_information=video_information)
    point_coords = image_slice.get_point_coordinates()
    point_labels = image_slice.centers_labels
    bounding_box = image_slice.get_box()

    ann_frame_idx = 0  # the frame index we interact with
    ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)

    points = point_coords.astype(np.float32)
    labels = point_labels.astype(np.int32)
    box = bounding_box.astype(np.float32)

    _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=ann_frame_idx,
        obj_id=ann_obj_id,
        points=points,
        labels=labels,
        box=box
    )

    # show the results on the current (interacted) frame
    if ShowPlots:
        plt.figure(figsize=(9, 6))
        plt.title(f'prompts')
        plt.imshow(Image.open(os.path.join(video_folder, frame_names[ann_frame_idx])))
        Debug.show_box(box, plt.gca())
        Debug.show_points(points, labels, plt.gca())
        Debug.show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])
        plt.tight_layout()
        plt.show()


def propagate_prompt_in_video(video_predictor: SAM2VideoPredictor, inference_state: dict) -> dict:
    logger.debug('propagate_prompt_in_video()')

    video_segments = {}  # video_segments contains the per-frame segmentation results
    for out_frame_idx, out_obj_ids, out_mask_logits in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    return video_segments


def save_raw_data(
        video_information: dict,
        frame_names: list,
        video_folder: Path,
        video_segments: dict,
        times: dict,
        timestamp: str
):
    logger.debug(f'save_raw_data()')

    seed_frame = video_information[SeedFrameKey]
    half_number = int(video_folder.stem[-1])

    # render the segmentation results every few frames
    vis_frame_stride = 1
    plt.close("all")
    raw_data = []
    for out_frame_idx in range(0, len(frame_names), vis_frame_stride):

        fake_seed = get_fake_seed(video_information)

        if fake_seed and out_frame_idx == 0:
            continue

        if ShowPlots:
            plt.figure(figsize=(6, 4))
            plt.title(f"frame {out_frame_idx}")
            plt.imshow(Image.open(os.path.join(video_folder, frame_names[out_frame_idx])))

        for out_obj_id, out_mask in video_segments[out_frame_idx].items():

            if ShowPlots:
                Debug.show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

            masks_file_path = Path(video_information[MasksKey])
            masks = load_masks(masks_file_path=masks_file_path)
            if half_number == 1:
                slice_number = seed_frame - out_frame_idx
            else:
                slice_number = seed_frame + out_frame_idx

            frame_number = out_frame_idx
            if fake_seed:
                if half_number == 1:
                    slice_number += 1
                else:
                    slice_number -= 1
                frame_number -= 1

            labeled_points = load_masks_slice(masks=masks, slice_number=slice_number)
            labels = np.unique(labeled_points)

            logger.debug(f'labels.size: {labels.size}')

            if labels.size > 1:
                jaccard, dice = compare_original_and_predicted_masks(
                    original_mask=labeled_points, predicted_mask=out_mask)

                raw_data_row = {
                    FrameKey: frame_number,
                    JaccardKey: jaccard,
                    DiceKey: dice,
                    SAMScoreKey: None,  # Can't find SAM 2 score, maybe the video predictor does no return one?
                    SetVideoTimeKey: times[SetVideoTimeKey],
                    PredictVideoTimeKey: times[PredictVideoTimeKey],
                    ProcessVideoTimeKey: times[ProcessVideoTimeKey],
                    SetImageTimeKey: times[SetImageTimeKey],
                    PredictImageTimeKey: times[PredictImageTimeKey],
                    ProcessImageTimeKey: times[ProcessImageTimeKey]
                }
            else:
                raw_data_row = {
                    FrameKey: out_frame_idx,
                    JaccardKey: None,
                    DiceKey: None,
                    SAMScoreKey: None,
                    SetVideoTimeKey: None,
                    PredictVideoTimeKey: None,
                    ProcessVideoTimeKey: None,
                    SetImageTimeKey: None,
                    PredictImageTimeKey: None,
                    ProcessImageTimeKey: None
                }
            raw_data.append(raw_data_row)

    output_folder = video_folder.parent
    output_file = output_folder / f'raw_data_half{half_number}_{timestamp}.csv'

    raw_data_df = pd.DataFrame(raw_data)
    raw_data_df.to_csv(output_file, index=False)


def process_video_half(
        dataset: Dataset,
        video_predictor: SAM2VideoPredictor,
        video_folder: Path,
        video_information: dict,
        timestamp: str
):
    frame_names = get_frame_names(video_folder)

    if ShowPlots:
        show_first_video_frame(
            video_folder=video_folder,
            video_information=video_information,
            frame_names=frame_names
        )

    set_video_start_time = perf_counter()
    inference_state = video_predictor.init_state(
        video_path=str(video_folder)
    )
    set_video_end_time = perf_counter()
    set_video_time = set_video_end_time - set_video_start_time
    set_image_time = set_video_time / len(frame_names)

    add_prompt_to_seed_frame(
        dataset=dataset,
        video_folder=video_folder,
        video_information=video_information,
        video_predictor=video_predictor,
        inference_state=inference_state,
        frame_names=frame_names
    )

    predict_video_start_time = perf_counter()
    video_segments = propagate_prompt_in_video(
        video_predictor=video_predictor,
        inference_state=inference_state
    )
    predict_video_end_time = perf_counter()

    predict_video_time = predict_video_end_time - predict_video_start_time
    predict_image_time = predict_video_time / len(frame_names)
    process_video_time = predict_video_end_time - set_video_start_time
    process_image_time = process_video_time / len(frame_names)

    times = {
        SetVideoTimeKey: set_video_time,
        PredictVideoTimeKey: predict_video_time,
        ProcessVideoTimeKey: process_video_time,
        SetImageTimeKey: set_image_time,
        PredictImageTimeKey: predict_image_time,
        ProcessImageTimeKey: process_image_time
    }

    save_raw_data(
        video_information=video_information,
        frame_names=frame_names,
        video_folder=video_folder,
        video_segments=video_segments,
        times=times,
        timestamp=timestamp
    )


def join_two_halves(
        video_folder: Path,
        timestamp: str
):
    raw_data_half1_df = pd.read_csv(video_folder / f'raw_data_half1_{timestamp}.csv')
    raw_data_half1_df = raw_data_half1_df[::-1]  # reverse the order of the rows
    raw_data_half1_df = raw_data_half1_df[:-1]  # remove the last row

    raw_data_half2_df = pd.read_csv(video_folder / f'raw_data_half2_{timestamp}.csv')

    raw_data_df = pd.concat([raw_data_half1_df, raw_data_half2_df])
    raw_data_df.reset_index(inplace=True)
    raw_data_df['index'] = raw_data_df.index
    raw_data_df.rename({'index': 'slice'}, axis=1, inplace=True)
    raw_data_df.drop(['frame'], axis=1, inplace=True)

    results_folder = video_folder.parent / 'results' / video_folder.stem
    if not results_folder.exists():
        results_folder.mkdir(parents=True)

    raw_data_df.to_csv(results_folder / f'raw_data_{timestamp}.csv', index=False)


def save_results(
        video_folder: Path,
        timestamp: str
):
    results_folder = video_folder.parent / 'results' / video_folder.stem
    raw_data_df = pd.read_csv(results_folder / f'raw_data_{timestamp}.csv')

    jaccard_column = raw_data_df[JaccardKey]
    jaccard_results = {
        MetricKey: JaccardKey,
        MinKey: jaccard_column.min(),
        MaxKey: jaccard_column.max(),
        AverageKey: jaccard_column.mean(),
        StandardDeviationKey: jaccard_column.std()
    }
    dice_column = raw_data_df[DiceKey]
    dice_results = {
        MetricKey: DiceKey,
        MinKey: dice_column.min(),
        MaxKey: dice_column.max(),
        AverageKey: dice_column.mean(),
        StandardDeviationKey: dice_column.std()
    }
    sam_score_column = raw_data_df[SAMScoreKey]
    sam_score_results = {
        MetricKey: SAMScoreKey,
        MinKey: sam_score_column.min(),
        MaxKey: sam_score_column.max(),
        AverageKey: sam_score_column.mean(),
        StandardDeviationKey: sam_score_column.std()
    }

    results = [jaccard_results, dice_results, sam_score_results]

    results_csv_output_path = results_folder / Path(f'results_{timestamp}.csv')
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_output_path, index=False)


def process_video(dataset: Dataset, video_predictor: SAM2VideoPredictor, video_folder: Path):
    timestamp = Timestamp.file()

    video_information = get_video_information(video_folder)

    video_halves = ['1', '2']

    items = len(video_halves)
    progress_bar_description = 'Processing video halves'
    progress_bar = tqdm(desc=progress_bar_description, total=items, leave=False, position=1)

    for video_half in video_halves:
        progress_bar.set_description(f'Processing video half {video_half}')

        process_video_half(
            dataset=dataset,
            video_predictor=video_predictor,
            video_folder=video_folder / f'half{video_half}',
            video_information=video_information,
            timestamp=timestamp
        )

        progress_bar.update()

    progress_bar.set_description(progress_bar_description)
    progress_bar.close()

    join_two_halves(video_folder=video_folder, timestamp=timestamp)
    save_results(video_folder=video_folder, timestamp=timestamp)


def main():
    if LoggingEnabled:
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")

    base_path = Path('working_data/covid/sam2_videos/')
    # base_path = Path('working_data/covid/sam2_videos_fake_seed/')
    video_folders = [
        base_path / 'image_coronacases_001/',
        base_path / 'image_coronacases_002/',
        base_path / 'image_coronacases_003/',
        base_path / 'image_coronacases_004/',
        base_path / 'image_coronacases_005/',
        base_path / 'image_coronacases_006/',
        base_path / 'image_coronacases_007/',
        base_path / 'image_coronacases_008/',
        base_path / 'image_coronacases_009/',
        base_path / 'image_coronacases_010/',
        base_path / 'image_coronacases_010/',
        base_path / 'image_radiopaedia_4_85506_1/',
        base_path / 'image_radiopaedia_7_85703_0/',
        base_path / 'image_radiopaedia_10_85902_1/',
        base_path / 'image_radiopaedia_10_85902_3/',
        base_path / 'image_radiopaedia_14_85914_0/',
        base_path / 'image_radiopaedia_27_86410_0/',
        base_path / 'image_radiopaedia_29_86490_1/',
        base_path / 'image_radiopaedia_29_86491_1/',
        base_path / 'image_radiopaedia_36_86526_0/',
        base_path / 'image_radiopaedia_40_86625_0/'
    ]

    video_predictor = get_sam2_video_predictor()

    items = len(video_folders)
    progress_bar_description = 'Processing videos'
    progress_bar = tqdm(desc=progress_bar_description, total=items, leave=True, position=0)

    for video_folder in video_folders:
        progress_bar.set_description(f'Processing video {video_folder.stem}')

        if Dataset.Coronacases.code in video_folder.stem:
            dataset = Dataset.Coronacases
        elif Dataset.Radiopaedia.code in video_folder.stem:
            dataset = Dataset.Radiopaedia
        else:
            raise ValueError(f'Unknown dataset: {video_folder.stem}')

        process_video(
            dataset=dataset,
            video_predictor=video_predictor,
            video_folder=video_folder
        )

        progress_bar.update()

    progress_bar.set_description(progress_bar_description)
    progress_bar.close()


if __name__ == '__main__':
    main()
