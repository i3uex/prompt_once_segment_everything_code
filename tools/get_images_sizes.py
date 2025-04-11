# Given a folder path, show a summary of the different sizes of the images
# in the folder (width x height: number of images).
from pathlib import Path

import pandas as pd
from PIL import Image
from rich import print
from tqdm import tqdm

SourceImagesPatterns = ['*.jpg', '*.png']


def get_images_sizes(folder_path: Path):
    images_paths = []
    for source_images_pattern in SourceImagesPatterns:
        images_paths += sorted(folder_path.glob(source_images_pattern))

    image_sizes = {'path': str(folder_path)}
    for images_path in images_paths:
        image = Image.open(images_path)
        image_size_key = f'{image.size[0]}x{image.size[1]}'
        if image_size_key in image_sizes:
            image_sizes[image_size_key] += 1
        else:
            image_sizes[image_size_key] = 1

    return image_sizes


def main():
    folder_paths = [
        'working_data/covid/sam2_videos/image_coronacases_001/all/',
        'working_data/covid/sam2_videos/image_coronacases_002/all/',
        'working_data/covid/sam2_videos/image_coronacases_003/all/',
        'working_data/covid/sam2_videos/image_coronacases_004/all/',
        'working_data/covid/sam2_videos/image_coronacases_005/all/',
        'working_data/covid/sam2_videos/image_coronacases_006/all/',
        'working_data/covid/sam2_videos/image_coronacases_007/all/',
        'working_data/covid/sam2_videos/image_coronacases_008/all/',
        'working_data/covid/sam2_videos/image_coronacases_009/all/',
        'working_data/covid/sam2_videos/image_coronacases_010/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_4_85506_1/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_7_85703_0/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_10_85902_1/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_10_85902_3/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_14_85914_0/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_27_86410_0/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_29_86490_1/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_29_86491_1/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_36_86526_0/all/',
        'working_data/covid/sam2_videos/image_radiopaedia_40_86625_0/all/',
        'datasets/montgomery/CXR_png/'
    ]

    progress_bar_description = 'Getting image sizes'
    items = len(folder_paths)
    progressbar = tqdm(desc=progress_bar_description, total=items)

    images_sizes = []
    for folder_path in folder_paths:
        progressbar.set_description(f'Getting image sizes for {folder_path}')
        image_sizes = get_images_sizes(Path(folder_path))
        images_sizes.append(image_sizes)

        progressbar.update()

    progressbar.set_description(progress_bar_description)
    progressbar.close()

    df = pd.DataFrame(images_sizes)
    df.to_csv('comparison/images_sizes.csv', index=False, float_format='%.0f')

    print(images_sizes)


if __name__ == '__main__':
    main()
