"""
Process a CT image or a slice of it. It performs the following steps:

1. If there are masks for the lungs:
    - find the centers of mass of each contour.
    - find the bounding boxes of each contour.
2. Use them as positive prompts:
    - "I'm looking for what this points mark."
    - "I'm looking for what's inside this box."
3. Use the center of the image as negative prompt ("This is the background").
4. Use SAM to segment the image using the provided prompts.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from rich import print
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from segment_anything import sam_model_registry, SamPredictor
from time import perf_counter
from tqdm import tqdm

from csv_keys import *
from sam_model import SamModel
from sam_prompt import SAMPrompt, SAMPromptJSONEncoder
from tools.argparse_helper import ArgumentParserHelper
from tools.debug import Debug
from tools.image_slice import ImageSlice
from tools.summarizer import Summarizer
from tools.timestamp import Timestamp

logger = logging.getLogger(__name__)

LoggingEnabled = True

DatasetPath = Path('datasets/covid')
DebugFolderPath = Path('debug')

DEBUG_DRAW_SAM_PREDICTION = bool(os.environ.get('DEBUG_DRAW_SAM_PREDICTION', 'True') == str(True))
DEBUG_DRAW_MASKS_CONTOURS = bool(os.environ.get('DEBUG_DRAW_MASKS_CONTOURS', 'True') == str(True))
DEBUG_DRAW_BOUNDING_BOX = bool(os.environ.get('DEBUG_DRAW_BOUNDING_BOX', 'True') == str(True))
DEBUG_DRAW_NEGATIVE_PROMPT = bool(os.environ.get('DEBUG_DRAW_NEGATIVE_PROMPT', 'True') == str(True))
DEBUG_INSERT_TITLE = bool(os.environ.get('DEBUG_INSERT_TITLE', 'True') == str(True))
USE_BOUNDING_BOX = bool(os.environ.get('USE_BOUNDING_BOX', 'True') == str(True))


# TODO: add documentation to this method, taken from SAM's notebooks.
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


# TODO: add documentation to this method, taken from SAM's notebooks.
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_points(coords, labels, ax, marker_size=375):
    positive_points = coords[labels == 1]
    negative_points = coords[labels == 0]

    # scatter shows x and y, but we are using rows and columns (y and x)
    rows = positive_points[:, 0]
    columns = positive_points[:, 1]
    ax.scatter(columns, rows, color='lime', marker='o', s=marker_size, edgecolor='white',
               linewidth=2.5)

    if DEBUG_DRAW_NEGATIVE_PROMPT:
        rows = negative_points[:, 0]
        columns = negative_points[:, 1]
        ax.scatter(columns, rows, color='red', marker='o', s=marker_size, edgecolor='black',
                   linewidth=2.5)


# TODO: add documentation to this method, taken from SAM's notebooks.
# https://github.com/facebookresearch/segment-anything/blob/main/notebooks/predictor_example.ipynb
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='orange', facecolor=(0, 0, 0, 0), lw=3))


def get_sam_predictor(sam_model: SamModel) -> SamPredictor:
    """
    Get an instance of the SAM predictor, given the model details.

    :param sam_model: model name and checkpoint to use.

    :return: an instance of the SAM predictor, given the model details.
    """

    logger.info('Get SAM predictor instance')
    logger.debug(f'get_sam_predictor('
                 f'sam_model={sam_model})')

    if sam_model.version == 1:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        logger.info(f'Device for SAM predictor: {device}')

        sam = sam_model_registry[sam_model.name](checkpoint=sam_model.checkpoint)
        sam.to(device)
        sam_predictor = SamPredictor(sam)
    elif sam_model.version == 2:
        device = 'cuda'
        torch.autocast(device_type=device, dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        sam2_model = build_sam2(sam_model.configuration, sam_model.checkpoint, device=device)
        sam_predictor = SAM2ImagePredictor(sam2_model)
    else:
        raise ValueError(f'Invalid SAM model version: {sam_model.version}')

    return sam_predictor


def load_image(image_file_path: Path) -> np.array:
    """
    Load the CT image.

    :param image_file_path: path of the image file.

    :return: CT image.
    """

    logger.info('Load the CT image')
    logger.debug(f'load_image('
                 f'image_file_path="{image_file_path}")')

    image_npz = np.load(str(image_file_path))
    image_npz_keys = list(image_npz.keys())
    image = image_npz[image_npz_keys[0]]

    return image


def load_masks(masks_file_path: Path) -> np.array:
    """
    Load the CT masks.

    :param masks_file_path: path of the masks file.

    :return: masks for the CT image.
    """

    logger.info('Load the CT image masks')
    logger.debug(f'load_masks('
                 f'masks_file_path="{masks_file_path}")')

    masks_npz = np.load(str(masks_file_path))
    masks_npz_keys = list(masks_npz.keys())
    masks = masks_npz[masks_npz_keys[0]]

    return masks


# TODO: these two methods could be the same.
def load_image_slice(image: np.array, slice_number: int) -> np.array:
    """
    Return a slice from a CT image, given its position. The slice is windowed
    to improve its contrast if needed, converted to greyscale, and expanded to
    RGB. It checks if the slice number exists.

    :param image: CT image from which to get the slice.
    :param slice_number: slice number to get from the image.

    :return: slice from a CT image.
    """

    logger.info('Load a slice from a CT image')
    logger.debug(f'load_image_slice('
                 f'image={image.shape}, '
                 f'slice_number={slice_number})')

    assert 0 <= slice_number < image.shape[-1]
    logger.info("Requested slice exists.")

    image_slice = image[:, :, slice_number]

    return image_slice


def load_masks_slice(masks: np.array, slice_number: int) -> np.array:
    """
    Return a slice masks from the list of masks, given its position. It checks
    if the slice number exists.

    :param masks: list of masks.
    :param slice_number: masks slice number to get from the list of masks.

    :return: masks slice from a list of masks.
    """

    logger.info('Load a masks slice from the list of masks')
    logger.debug(f'load_masks_slice('
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number})')

    assert 0 <= slice_number < masks.shape[-1]
    logger.info("Requested masks slice exists.")

    masks_slice = masks[:, :, slice_number]

    return masks_slice


def compare_original_and_predicted_masks(
        original_mask: np.array, predicted_mask: np.array
) -> Tuple[float, float]:
    """
    Compares the original segmentation mask with the one predicted. Returns a
    tuple with the Jaccard index and the Dice coefficient.

    :param original_mask: original segmentation mask.
    :param predicted_mask: predicted segmentation mask.

    :return: Jaccard index and the Dice coefficient of the masks provided.
    """

    logger.info('Compare original and predicted masks')
    logger.debug(f'compare_original_and_predicted_masks('
                 f'original_mask={original_mask.shape}, '
                 f'predicted_mask={predicted_mask.shape})')

    original_mask_as_bool = original_mask != 0
    predicted_mask_transformed = np.squeeze(predicted_mask)

    intersection = original_mask_as_bool * predicted_mask_transformed
    union = (original_mask_as_bool + predicted_mask_transformed) > 0

    jaccard = intersection.sum() / float(union.sum())
    dice = intersection.sum() * 2 / (original_mask_as_bool.sum() + predicted_mask.sum())

    return jaccard, dice


def save_results(output_path: Path, list_of_dictionaries: list) -> Tuple[Path, Path]:
    """
    Save the result to a CSV file.

    :param output_path: where the results must be saved.
    :param list_of_dictionaries: results to save.

    :return: Paths to the resulting CSV files.
    """

    logger.info('Save results')
    logger.debug(f'save_result('
                 f'output_path={output_path}, '
                 f'list_of_dictionaries={list_of_dictionaries})')

    timestamp = Timestamp.file()

    output_path.mkdir(parents=True, exist_ok=True)

    df_raw_data = pd.DataFrame(list_of_dictionaries)

    # Save results
    jaccard_column = df_raw_data[JaccardKey]
    jaccard_results = {
        MetricKey: JaccardKey,
        MinKey: jaccard_column.min(),
        MaxKey: jaccard_column.max(),
        AverageKey: jaccard_column.mean(),
        StandardDeviationKey: jaccard_column.std()
    }
    dice_column = df_raw_data[DiceKey]
    dice_results = {
        MetricKey: DiceKey,
        MinKey: dice_column.min(),
        MaxKey: dice_column.max(),
        AverageKey: dice_column.mean(),
        StandardDeviationKey: dice_column.std()
    }
    sam_score_column = df_raw_data[SAMScoreKey]
    sam_score_results = {
        MetricKey: SAMScoreKey,
        MinKey: sam_score_column.min(),
        MaxKey: sam_score_column.max(),
        AverageKey: sam_score_column.mean(),
        StandardDeviationKey: sam_score_column.std()
    }

    results = [jaccard_results, dice_results, sam_score_results]

    results_csv_output_path = output_path / Path(f'results_{timestamp}.csv')
    df_results = pd.DataFrame(results)
    df_results.to_csv(results_csv_output_path, index=False)

    # Save raw data
    raw_data_csv_output_path = output_path / Path(f'raw_data_{timestamp}.csv')
    df_raw_data.to_csv(raw_data_csv_output_path, index=False)

    return results_csv_output_path, raw_data_csv_output_path


def save_sam_prompts(
        output_path: Path,
        sam_prompts: list
) -> Path:
    """
    Save the list of SAM prompts to file in JSON format.

    :param output_path: where the results must be saved.
    :param sam_prompts: SAM prompts to save.

    :return: Path to the resulting file.
    """

    logger.info('Save SAM prompts')
    logger.debug(f'save_sam_prompts('
                 f'output_path={output_path}, '
                 f'sam_prompts={sam_prompts})')

    timestamp = Timestamp.file()

    output_path.mkdir(parents=True, exist_ok=True)

    # Save prompts
    sam_prompts_output_path = output_path / Path(f'sam_prompts_{timestamp}.json')
    with open(sam_prompts_output_path, 'w') as output_file:
        json.dump(
            sam_prompts,
            output_file,
            cls=SAMPromptJSONEncoder,
            sort_keys=False,
            indent=4,
            ensure_ascii=True)

    # TODO: propagate slice=None (only tested with a slice number, 0 for none).
    # TODO: draw points and bounding box on slice to check if it's correct.

    return sam_prompts_output_path


def process_image_slice(sam_predictor: SamPredictor,
                        image: np.array,
                        masks: np.array,
                        slice_number: int,
                        apply_windowing: bool,
                        use_masks_contours: bool,
                        use_bounding_box: bool,
                        multimask_output: bool,
                        debug: Debug) -> Tuple[dict, SAMPrompt]:
    """
    Process a slice of the image. Returns the result of the analysis.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param slice_number: slice to work with.
    :param apply_windowing: if True, apply windowing to the image.
    :param use_masks_contours: if True, get positive prompts from contours.
    :param use_bounding_box: if True, include a bounding box in the prompts.
    :param multimask_output: if True, let SAM return multiple masks.
    :param debug: instance of Debug class.

    :return: a tuple with two dictionaries. The first, with the number of the
    slice been processed and the Jaccard index and Dice score between the
    ground truth and the prediction masks. The second, with the prompt provided
    to SAM, so it can perform the segmentation.
    """

    logger.info('Process image slice')
    logger.debug(f'process_image_slice('
                 f'sam_predictor={sam_predictor.device.type}, '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number}, '
                 f'apply_windowing={apply_windowing}, '
                 f'use_masks_contours={use_masks_contours}, '
                 f'use_bounding_box={use_bounding_box}, '
                 f'multimask_output={multimask_output}, '
                 f'debug={debug.enabled})')

    points = load_image_slice(image=image, slice_number=slice_number)
    labeled_points = load_masks_slice(masks=masks, slice_number=slice_number)

    image_slice = ImageSlice(
        points=points,
        labeled_points=labeled_points,
        apply_windowing=apply_windowing,
        use_bounding_box=use_bounding_box,
        use_masks_contours=use_masks_contours)

    mask = []
    score = []
    jaccard = None
    dice = None
    sam_score = None
    set_image_time = None
    predict_image_time = None
    process_image_time = None

    if image_slice.labels.size > 1:
        point_coords = image_slice.get_point_coordinates()
        point_labels = image_slice.centers_labels
        if use_bounding_box:
            bounding_box = image_slice.get_box()
        else:
            bounding_box = None

        set_image_start_time = perf_counter()
        sam_predictor.set_image(image_slice.processed_points)
        set_image_end_time = perf_counter()

        predict_image_start_time = perf_counter()
        if USE_BOUNDING_BOX:
            mask, score, logits = sam_predictor.predict(
                point_coords=point_coords.copy(),
                point_labels=point_labels,
                box=bounding_box,
                multimask_output=multimask_output)
        else:
            mask, score, logits = sam_predictor.predict(
                point_coords=point_coords.copy(),
                point_labels=point_labels,
                multimask_output=multimask_output)
        predict_image_end_time = perf_counter()

        set_image_time = set_image_end_time - set_image_start_time
        predict_image_time = predict_image_end_time - predict_image_start_time
        process_image_time = predict_image_end_time - set_image_start_time

        if multimask_output:
            max_index = np.argmax(score)
            mask = mask[[max_index], :, :]
            score = score[max_index]

        sam_prompt = SAMPrompt(
            image_file_path=debug.image_file_path,
            masks_file_path=debug.masks_file_path,
            slice_number=slice_number,
            points_cords=point_coords,
            points_labels=point_labels,
            bounding_box=bounding_box
        )

        # Compare original and predicted lung masks
        jaccard, dice = compare_original_and_predicted_masks(
            original_mask=labeled_points, predicted_mask=mask)
        if not multimask_output:
            sam_score = score[0]
        else:
            sam_score = np.max(score)
    else:
        logger.info("There are no masks for the current slice")
        sam_prompt = None

    if debug.enabled:
        if image_slice.labels.size > 1:
            debug.set_slice_number(slice_number=slice_number)

            # Save SAM's prompt to YML
            prompts = dict()
            for index, contour_center in enumerate(image_slice.centers):
                row = int(contour_center[0])
                column = int(contour_center[1])
                label = int(image_slice.centers_labels[index])
                prompts.update({
                    index: {
                        'row': row,
                        'column': column,
                        'label': label
                    }
                })

            if use_bounding_box:
                bounding_box = image_slice.get_box()
                prompts.update({
                    'bounding_box': {
                        'row_min': int(bounding_box[1]),
                        'colum_min': int(bounding_box[0]),
                        'row_max': int(bounding_box[3]),
                        'column_max': int(bounding_box[2])
                    }
                })

            data = dict(
                image=debug.image_file_path.name,
                masks=debug.masks_file_path.name,
                slice=slice_number,
                prompts=prompts
            )

            debug_file_path = debug.get_file_path('prompt', '.yml')
            with open(debug_file_path, 'w') as file:
                yaml.dump(data, file, sort_keys=False)

            # Save SAM segmentation
            figure = plt.figure(figsize=(10, 10))
            plt.imshow(image_slice.processed_points)
            if DEBUG_DRAW_SAM_PREDICTION:
                show_mask(mask, plt.gca())
            if DEBUG_DRAW_MASKS_CONTOURS:
                for mask_contour in image_slice.contours:
                    plt.plot(mask_contour[:, 1], mask_contour[:, 0], color='green', zorder=0)
            show_points(
                coords=image_slice.centers,
                labels=image_slice.centers_labels,
                ax=plt.gca())
            if DEBUG_DRAW_BOUNDING_BOX:
                if use_bounding_box:
                    show_box(box=image_slice.get_box(), ax=plt.gca())
            if DEBUG_INSERT_TITLE:
                plt.title(f"Score: {score[0]:.3f}", fontsize=18)
            plt.axis('off')

            debug_file_path = debug.get_file_path('prediction', '.png')
            figure.savefig(debug_file_path, bbox_inches='tight', pad_inches=0)
            plt.close()

    result = {
        SliceNumberKey: slice_number,
        JaccardKey: jaccard,
        DiceKey: dice,
        SAMScoreKey: sam_score,
        SetImageTimeKey: set_image_time,
        PredictImageTimeKey: predict_image_time,
        ProcessImageTimeKey: process_image_time
    }

    return result, sam_prompt


def process_image(sam_predictor: SamPredictor,
                  image: np.array,
                  masks: np.array,
                  apply_windowing: bool,
                  use_bounding_box: bool,
                  use_masks_contours: bool,
                  multimask_output: bool,
                  debug: Debug) -> Tuple[Path, Path, Path]:
    """
    Process all the slices of a given image. Saves the result as two CSV files,
    one with each slice's result, another with a statistical summary. Returns
    the paths where the resulting CSV files will be stored.

    :param sam_predictor: SAM predictor for image segmentation.
    :param image: array with the slices of the CT.
    :param masks: masks for each slice of the CT.
    :param apply_windowing: if True, apply windowing to the image.
    :param use_masks_contours: if True, get positive prompts from contours.
    :param use_bounding_box: if True, include a bounding box in the prompts.
    :param multimask_output: if True, let SAM return multiple masks.
    :param debug: instance of Debug class.

    :return: paths where the resulting files are stored.
    """

    logger.info('Process image')
    logger.debug(f'process_image('
                 f'sam_predictor={sam_predictor.device.type}, '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'apply_windowing={apply_windowing}, '
                 f'use_masks_contours={use_masks_contours}, '
                 f'use_bounding_box={use_bounding_box}, '
                 f'multimask_output={multimask_output}, '
                 f'debug={debug.enabled})')

    items = image.shape[-1]
    progress_bar = tqdm(desc='Processing CT image slices', total=items)

    results = []
    sam_prompts = []
    for slice_number in range(items):
        result, sam_prompt = process_image_slice(sam_predictor=sam_predictor,
                                                 image=image,
                                                 masks=masks,
                                                 slice_number=slice_number,
                                                 apply_windowing=apply_windowing,
                                                 use_masks_contours=use_masks_contours,
                                                 use_bounding_box=use_bounding_box,
                                                 multimask_output=multimask_output,
                                                 debug=debug)
        results.append(result)
        if sam_prompt is not None:
            sam_prompts.append(sam_prompt)
        progress_bar.update()
    progress_bar.close()

    output_path = debug.image_file_path.parent / Path('results') / Path(debug.image_file_path.stem)
    results_path, raw_data_path = save_results(output_path, results)
    sam_prompts_path = save_sam_prompts(output_path, sam_prompts)

    return results_path, raw_data_path, sam_prompts_path


def parse_arguments() -> Tuple[int, Path, Path, int, bool, bool, bool, bool, bool, bool]:
    """
    Parse arguments passed via command line, returning them formatted. Adequate
    defaults are provided when possible.

    :return: SAM version to use for segmentation (1 or 2), path of the image
    file, path of the masks file, slice to work with, perform windowing on the
    image slice, use mask contours to get the positive point prompts, use a
    bounding box as a prompt, let SAM return multiple masks, dry run option,
    debug option.
    """

    logger.info('Get script arguments')
    logger.debug('parse_arguments()')

    program_description = 'Process image'
    argument_parser = argparse.ArgumentParser(description=program_description)
    argument_parser.add_argument('-v', '--sam_version',
                                 type=int, default=1,
                                 help='SAM model version')
    argument_parser.add_argument('-i', '--image_file_path',
                                 required=True,
                                 help='path to image file')
    argument_parser.add_argument('-m', '--masks_file_path',
                                 required=True,
                                 help='path to masks file')
    argument_parser.add_argument('-s', '--slice',
                                 required=False,
                                 help='slice to work with')
    argument_parser.add_argument('-w', '--apply_windowing',
                                 action='store_true',
                                 help='apply windowing to the image')
    argument_parser.add_argument('-c', '--use_masks_contours',
                                 action='store_true',
                                 help='get positive prompts from contours')
    argument_parser.add_argument('-b', '--use_bounding_box',
                                 action='store_true',
                                 help='include a bounding box in the prompts')
    argument_parser.add_argument('-o', '--multimask_output',
                                 action='store_true',
                                 help='enable multiple masks')
    argument_parser.add_argument('-n', '--dry_run',
                                 action='store_true',
                                 help='show what would be done, do not do it')
    argument_parser.add_argument('-d', '--debug',
                                 action='store_true',
                                 help='save debug data for later inspection')

    arguments = argument_parser.parse_args()
    sam_version = arguments.sam_version
    image_file_path = ArgumentParserHelper.parse_file_path_segment(
        arguments.image_file_path)
    masks_file_path = ArgumentParserHelper.parse_file_path_segment(
        arguments.masks_file_path)
    if arguments.slice is not None:
        slice_number = ArgumentParserHelper.parse_integer(arguments.slice)
    else:
        slice_number = None
    apply_windowing = arguments.apply_windowing
    use_masks_contours = arguments.use_masks_contours
    use_bounding_box = arguments.use_bounding_box
    multimask_output = arguments.multimask_output
    dry_run = arguments.dry_run
    debug = arguments.debug

    return sam_version, Path(image_file_path), Path(masks_file_path), \
        slice_number, apply_windowing, use_masks_contours, use_bounding_box, multimask_output, \
        dry_run, debug


def get_summary(
        sam_model: SamModel,
        image_file_path: Path,
        masks_file_path: Path,
        image: np.array,
        masks: np.array,
        slice_number: int,
        apply_windowing: bool,
        use_masks_contours: bool,
        use_bounding_box: bool,
        multimask_output: bool,
        dry_run: bool,
        debug: Debug
) -> str:
    """
    Show a summary of the actions this script will perform.

    :param sam_model: details of the model to use.
    :param image_file_path: path of the image file.
    :param masks_file_path: path of the masks file.
    :param image: array with the CT volume slices.
    :param masks: array with the CT volume masks.
    :param slice_number: slice to work with.
    :param apply_windowing: if True, apply windowing to the image.
    :param use_masks_contours: if True, get positive prompts from contours.
    :param use_bounding_box: if True, include a bounding box in the prompts.
    :param multimask_output: if True, let SAM return multiple masks.
    :param dry_run: if True, the actions will not be performed.
    :param debug: instance of Debug class.

    :return: summary of the actions this script will perform.
    """

    logger.info('Get summary')
    logger.debug(f'get_summary('
                 f'sam_model={sam_model}, '
                 f'image_file_path="{image_file_path}", '
                 f'masks_file_path="{masks_file_path}", '
                 f'image={image.shape}, '
                 f'masks={masks.shape}, '
                 f'slice_number={slice_number}, '
                 f'apply_windowing={apply_windowing}, '
                 f'use_masks_contours={use_masks_contours}, '
                 f'use_bounding_box={use_bounding_box}, '
                 f'multimask_output={multimask_output}, '
                 f'debug={debug}, '
                 f'dry_run={dry_run})')

    image_slices = image.shape[-1]

    masks_slices = masks.shape[-1]

    if slice_number is not None:
        requested_slice_in_range = slice_number < image_slices
        slice_information = f'Slice: {slice_number}'
    else:
        requested_slice_in_range = None
        slice_information = 'Process all slices'

    summary = f'- Model: "{sam_model.description}"\n' \
              f'- Image file path: "{image_file_path}"\n' \
              f'- Masks file path: "{masks_file_path}"\n' \
              f'- {slice_information}\n' \
              f'- Apply windowing: {apply_windowing}\n' \
              f'- Use masks contours: {use_masks_contours}\n' \
              f'- Use bounding box: {use_bounding_box}\n' \
              f'- Let SAM return multiple masks: {multimask_output}\n' \
              f'- Debug: {debug.enabled}\n' \
              f'- Dry run: {dry_run}\n' \
              f'- Image slices: {image_slices}\n' \
              f'- Masks slices: {masks_slices}\n' \
              f'- Equal number of slices: {image_slices == masks_slices}'

    if requested_slice_in_range is not None:
        summary += f'\n' \
                   f'- Requested slice in range: {requested_slice_in_range}'

    return summary


def main():
    """
    Set logging up, parse arguments, and process data.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.colorbar").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True

    logger.info("Start processing data")
    logger.debug("main()")

    summarizer = Summarizer()

    sam_version, image_file_path, masks_file_path, slice_number, \
        apply_windowing, use_masks_contours, use_bounding_box, multimask_output, \
        dry_run, debug_enabled = parse_arguments()

    if sam_version == 1:
        sam_model = SamModel.ViT_L
    elif sam_version == 2:
        sam_model = SamModel.Large
    else:
        raise ValueError(f'Invalid SAM version: {sam_version}')

    debug = Debug(
        enabled=debug_enabled,
        image_file_path=image_file_path,
        masks_file_path=masks_file_path)

    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    summarizer.summary = get_summary(
        sam_model=sam_model,
        image_file_path=image_file_path,
        masks_file_path=masks_file_path,
        image=image,
        masks=masks,
        slice_number=slice_number,
        apply_windowing=apply_windowing,
        use_masks_contours=use_masks_contours,
        use_bounding_box=use_bounding_box,
        multimask_output=multimask_output,
        debug=debug,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    sam_predictor = get_sam_predictor(sam_model)

    if slice_number is None:
        result = process_image(sam_predictor=sam_predictor,
                               image=image,
                               masks=masks,
                               apply_windowing=apply_windowing,
                               use_masks_contours=use_masks_contours,
                               use_bounding_box=use_bounding_box,
                               multimask_output=multimask_output,
                               debug=debug)
        print(f'Results saved to: "{str(result[0])}"')
        print(f'Raw data saved to: "{str(result[1])}"')
    else:
        result = process_image_slice(sam_predictor=sam_predictor,
                                     image=image,
                                     masks=masks,
                                     slice_number=slice_number,
                                     apply_windowing=apply_windowing,
                                     use_masks_contours=use_masks_contours,
                                     use_bounding_box=use_bounding_box,
                                     multimask_output=multimask_output,
                                     debug=debug)
        print(f'Results saved to: "{str(result[0])}"')
        print(f'Prompt saved to: "{str(result[1])}"')
        print(f'Jaccard index: {result[0][JaccardKey]:.4f}')
        print(f'Dice score: {result[0][DiceKey]:.4f}')

    print(summarizer.notification_message)


# Loads the same Montgomery image and masks 250 times, to test the performance
# of the SAM models.
def main_montgomery_1_by_250():
    """
    Set logging up, parse arguments, and process data.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.colorbar").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True

    logger.info("Start processing data")
    logger.debug("main()")

    summarizer = Summarizer()

    sam_version, image_file_path, masks_file_path, slice_number, \
        apply_windowing, use_masks_contours, use_bounding_box, multimask_output, \
        dry_run, debug_enabled = parse_arguments()

    if sam_version == 1:
        sam_model = SamModel.ViT_L
    elif sam_version == 2:
        sam_model = SamModel.Large
    else:
        raise ValueError(f'Invalid SAM version: {sam_version}')

    debug = Debug(
        enabled=debug_enabled,
        image_file_path=image_file_path,
        masks_file_path=masks_file_path)

    image = load_image(image_file_path=image_file_path)
    masks = load_masks(masks_file_path=masks_file_path)

    summarizer.summary = get_summary(
        sam_model=sam_model,
        image_file_path=image_file_path,
        masks_file_path=masks_file_path,
        image=image,
        masks=masks,
        slice_number=slice_number,
        apply_windowing=apply_windowing,
        use_masks_contours=use_masks_contours,
        use_bounding_box=use_bounding_box,
        multimask_output=multimask_output,
        debug=debug,
        dry_run=dry_run)

    if dry_run:
        print()
        print('[bold]Summary:[/bold]')
        print(summarizer.summary)
        return

    sam_predictor = get_sam_predictor(sam_model)

    if slice_number is None:
        for iteration in range(250):
            print(f'Iteration: {iteration}/250')
            result = process_image(sam_predictor=sam_predictor,
                                   image=image,
                                   masks=masks,
                                   apply_windowing=apply_windowing,
                                   use_masks_contours=use_masks_contours,
                                   use_bounding_box=use_bounding_box,
                                   multimask_output=multimask_output,
                                   debug=debug)
            print(f'Results saved to: "{str(result[0])}"')
            print(f'Raw data saved to: "{str(result[1])}"')
    else:
        result = process_image_slice(sam_predictor=sam_predictor,
                                     image=image,
                                     masks=masks,
                                     slice_number=slice_number,
                                     apply_windowing=apply_windowing,
                                     use_masks_contours=use_masks_contours,
                                     use_bounding_box=use_bounding_box,
                                     multimask_output=multimask_output,
                                     debug=debug)
        print(f'Results saved to: "{str(result[0])}"')
        print(f'Prompt saved to: "{str(result[1])}"')
        print(f'Jaccard index: {result[0][JaccardKey]:.4f}')
        print(f'Dice score: {result[0][DiceKey]:.4f}')

    print(summarizer.notification_message)


# Processes all the Montgomery images, but the model is loaded only once.
def main_montgomery():
    """
    Set logging up, parse arguments, and process data.
    """

    if LoggingEnabled:
        logging.basicConfig(
            filename="debug.log",
            level=logging.DEBUG,
            format="%(asctime)-15s %(levelname)8s %(name)s %(message)s")
        logging.getLogger("matplotlib.font_manager").disabled = True
        logging.getLogger("matplotlib.colorbar").disabled = True
        logging.getLogger("matplotlib.pyplot").disabled = True

    logger.info("Start processing data")
    logger.debug("main()")

    summarizer = Summarizer()

    sam_version, image_file_path, masks_file_path, slice_number, \
        apply_windowing, use_masks_contours, use_bounding_box, multimask_output, \
        dry_run, debug_enabled = parse_arguments()

    if sam_version == 1:
        sam_model = SamModel.ViT_L
    elif sam_version == 2:
        sam_model = SamModel.Large
    else:
        raise ValueError(f'Invalid SAM version: {sam_version}')

    sam_predictor = get_sam_predictor(sam_model)

    image_items = [
        'MCUCXR_0001_0',
        'MCUCXR_0002_0',
        'MCUCXR_0003_0',
        'MCUCXR_0004_0',
        'MCUCXR_0005_0',
        'MCUCXR_0006_0',
        'MCUCXR_0008_0',
        'MCUCXR_0011_0',
        'MCUCXR_0013_0',
        'MCUCXR_0015_0',
        'MCUCXR_0016_0',
        'MCUCXR_0017_0',
        'MCUCXR_0019_0',
        'MCUCXR_0020_0',
        'MCUCXR_0021_0',
        'MCUCXR_0022_0',
        'MCUCXR_0023_0',
        'MCUCXR_0024_0',
        'MCUCXR_0026_0',
        'MCUCXR_0027_0',
        'MCUCXR_0028_0',
        'MCUCXR_0029_0',
        'MCUCXR_0030_0',
        'MCUCXR_0031_0',
        'MCUCXR_0035_0',
        'MCUCXR_0038_0',
        'MCUCXR_0040_0',
        'MCUCXR_0041_0',
        'MCUCXR_0042_0',
        'MCUCXR_0043_0',
        'MCUCXR_0044_0',
        'MCUCXR_0045_0',
        'MCUCXR_0046_0',
        'MCUCXR_0047_0',
        'MCUCXR_0048_0',
        'MCUCXR_0049_0',
        'MCUCXR_0051_0',
        'MCUCXR_0052_0',
        'MCUCXR_0053_0',
        'MCUCXR_0054_0',
        'MCUCXR_0055_0',
        'MCUCXR_0056_0',
        'MCUCXR_0057_0',
        'MCUCXR_0058_0',
        'MCUCXR_0059_0',
        'MCUCXR_0060_0',
        'MCUCXR_0061_0',
        'MCUCXR_0062_0',
        'MCUCXR_0063_0',
        'MCUCXR_0064_0',
        'MCUCXR_0068_0',
        'MCUCXR_0069_0',
        'MCUCXR_0070_0',
        'MCUCXR_0071_0',
        'MCUCXR_0072_0',
        'MCUCXR_0074_0',
        'MCUCXR_0075_0',
        'MCUCXR_0077_0',
        'MCUCXR_0079_0',
        'MCUCXR_0080_0',
        'MCUCXR_0081_0',
        'MCUCXR_0082_0',
        'MCUCXR_0083_0',
        'MCUCXR_0084_0',
        'MCUCXR_0085_0',
        'MCUCXR_0086_0',
        'MCUCXR_0087_0',
        'MCUCXR_0089_0',
        'MCUCXR_0090_0',
        'MCUCXR_0091_0',
        'MCUCXR_0092_0',
        'MCUCXR_0094_0',
        'MCUCXR_0095_0',
        'MCUCXR_0096_0',
        'MCUCXR_0097_0',
        'MCUCXR_0099_0',
        'MCUCXR_0100_0',
        'MCUCXR_0101_0',
        'MCUCXR_0102_0',
        'MCUCXR_0103_0',
        'MCUCXR_0104_1',
        'MCUCXR_0108_1',
        'MCUCXR_0113_1',
        'MCUCXR_0117_1',
        'MCUCXR_0126_1',
        'MCUCXR_0140_1',
        'MCUCXR_0141_1',
        'MCUCXR_0142_1',
        'MCUCXR_0144_1',
        'MCUCXR_0150_1',
        'MCUCXR_0162_1',
        'MCUCXR_0166_1',
        'MCUCXR_0170_1',
        'MCUCXR_0173_1',
        'MCUCXR_0182_1',
        'MCUCXR_0188_1',
        'MCUCXR_0194_1',
        'MCUCXR_0195_1',
        'MCUCXR_0196_1',
        'MCUCXR_0203_1',
        'MCUCXR_0213_1',
        'MCUCXR_0218_1',
        'MCUCXR_0223_1',
        'MCUCXR_0228_1',
        'MCUCXR_0243_1',
        'MCUCXR_0251_1',
        'MCUCXR_0253_1',
        'MCUCXR_0254_1',
        'MCUCXR_0255_1',
        'MCUCXR_0258_1',
        'MCUCXR_0264_1',
        'MCUCXR_0266_1',
        'MCUCXR_0275_1',
        'MCUCXR_0282_1',
        'MCUCXR_0289_1',
        'MCUCXR_0294_1',
        'MCUCXR_0301_1',
        'MCUCXR_0309_1',
        'MCUCXR_0311_1',
        'MCUCXR_0313_1',
        'MCUCXR_0316_1',
        'MCUCXR_0331_1',
        'MCUCXR_0334_1',
        'MCUCXR_0338_1',
        'MCUCXR_0348_1',
        'MCUCXR_0350_1',
        'MCUCXR_0352_1',
        'MCUCXR_0354_1',
        'MCUCXR_0362_1',
        'MCUCXR_0367_1',
        'MCUCXR_0369_1',
        'MCUCXR_0372_1',
        'MCUCXR_0375_1',
        'MCUCXR_0383_1',
        'MCUCXR_0387_1',
        'MCUCXR_0390_1',
        'MCUCXR_0393_1',
        'MCUCXR_0399_1'
    ]

    items = len(image_items)
    progress_bar = tqdm(desc='Processing Montgomery images', total=items, leave=True, position=0)

    for image_item in image_items:
        progress_bar.set_description(f'Processing {image_item}')

        image_file_path = Path(f'working_data/montgomery/image_{image_item}.npz')
        masks_file_path = Path(f'working_data/montgomery/masks_{image_item}.npz')

        image = load_image(image_file_path=image_file_path)
        masks = load_masks(masks_file_path=masks_file_path)

        debug = Debug(
            enabled=debug_enabled,
            image_file_path=image_file_path,
            masks_file_path=masks_file_path)

        summarizer.summary = get_summary(
            sam_model=sam_model,
            image_file_path=image_file_path,
            masks_file_path=masks_file_path,
            image=image,
            masks=masks,
            slice_number=slice_number,
            apply_windowing=apply_windowing,
            use_masks_contours=use_masks_contours,
            use_bounding_box=use_bounding_box,
            multimask_output=multimask_output,
            debug=debug,
            dry_run=dry_run)

        if dry_run:
            print()
            print('[bold]Summary:[/bold]')
            print(summarizer.summary)
            return

        result = process_image(sam_predictor=sam_predictor,
                               image=image,
                               masks=masks,
                               apply_windowing=apply_windowing,
                               use_masks_contours=use_masks_contours,
                               use_bounding_box=use_bounding_box,
                               multimask_output=multimask_output,
                               debug=debug)
        print(f'Results saved to: "{str(result[0])}"')
        print(f'Raw data saved to: "{str(result[1])}"')

        print(summarizer.notification_message)

        progress_bar.update()

    progress_bar.set_description('Processing Montgomery images')
    progress_bar.close()


if __name__ == '__main__':
    main()
    # main_montgomery_1_by_250()
    # main_montgomery()
