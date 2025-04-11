from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from data_model.dataset import Dataset
from tools.files import get_most_recent_timestamped_file

PYPLOT_PLOT_WIDTH = 1280
PYPLOT_PLOT_HEIGHT = 960
PYPLOT_PLOT_DPI = 100
PYPLOT_SIZE = 22
PYPLOT_FONT_FAMILY = 'serif'
PYPLOT_LEGEND_FONT_SIZE = 'large'
PYPLOT_LEGEND_FACE_COLOR = 'white'
PYPLOT_FIGURE_SIZE = (10, 8)
PYPLOT_AXES_LABEL_SIZE = PYPLOT_SIZE
PYPLOT_AXES_TITLE_SIZE = PYPLOT_SIZE
PYPLOT_GRID_COLOR = 'white'
PYPLOT_XTICK_LABEL_SIZE = PYPLOT_SIZE * 0.85
PYPLOT_YTICK_LABEL_SIZE = PYPLOT_SIZE * 0.85
PYPLOT_TICK_FONT_COLOR = '#535353'
PYPLOT_AXES_TITLE_PAD = 25
PYPLOT_AXES_EDGE_COLOR = 'white'
PYPLOT_AXES_FACE_COLOR = '#e5e5e5'
PYPLOT_FIGURE_FACECOLOR = 'white'

PyPlotParameters = {
    'font.family': PYPLOT_FONT_FAMILY,
    'legend.fontsize': PYPLOT_LEGEND_FONT_SIZE,
    'legend.facecolor': PYPLOT_LEGEND_FACE_COLOR,
    'figure.figsize': PYPLOT_FIGURE_SIZE,
    'axes.labelsize': PYPLOT_AXES_LABEL_SIZE,
    'axes.titlesize': PYPLOT_AXES_TITLE_SIZE,
    'grid.color': PYPLOT_GRID_COLOR,
    'xtick.labelsize': PYPLOT_XTICK_LABEL_SIZE,
    'ytick.labelsize': PYPLOT_YTICK_LABEL_SIZE,
    'axes.titlepad': PYPLOT_AXES_TITLE_PAD,
    'axes.edgecolor': PYPLOT_AXES_EDGE_COLOR,
    'axes.facecolor': PYPLOT_AXES_FACE_COLOR,
    'xtick.color': PYPLOT_TICK_FONT_COLOR,
    'ytick.color': PYPLOT_TICK_FONT_COLOR,
    'figure.facecolor': PYPLOT_FIGURE_FACECOLOR,
    'axes.formatter.use_locale': True
}
RawDataFilePattern = 'joint_raw_data_*.csv'


def load_data(results_folder_path: Path) -> pd.DataFrame:
    file_path = get_most_recent_timestamped_file(
        results_folder_path, RawDataFilePattern)
    df_all = pd.read_csv(file_path)
    df_all.rename(columns={
        'jaccard': 'jaccard_all',
        'dice': 'dice_all',
        'sam_score': 'sam_score_all'
    }, inplace=True)
    return df_all


def get_sam_1_vs_sam_2_dataframe(df_all_sam_1: pd.DataFrame, df_all_sam_2: pd.DataFrame) -> pd.DataFrame:
    jaccard_index_minimum_sam_1 = df_all_sam_1['jaccard_all'].min()
    jaccard_index_minimum_sam_2 = df_all_sam_2['jaccard_all'].min()
    jaccard_index_minimum_delta = jaccard_index_minimum_sam_2 - jaccard_index_minimum_sam_1

    jaccard_index_maximum_sam_1 = df_all_sam_1['jaccard_all'].max()
    jaccard_index_maximum_sam_2 = df_all_sam_2['jaccard_all'].max()
    jaccard_index_maximum_delta = jaccard_index_maximum_sam_2 - jaccard_index_maximum_sam_1

    jaccard_index_average_sam_1 = df_all_sam_1['jaccard_all'].mean()
    jaccard_index_average_sam_2 = df_all_sam_2['jaccard_all'].mean()
    jaccard_index_average_delta = jaccard_index_average_sam_2 - jaccard_index_average_sam_1

    jaccard_index_standard_deviation_sam_1 = df_all_sam_1['jaccard_all'].std()
    jaccard_index_standard_deviation_sam_2 = df_all_sam_2['jaccard_all'].std()
    jaccard_index_standard_deviation_delta = jaccard_index_standard_deviation_sam_2 - jaccard_index_standard_deviation_sam_1

    dice_score_minimum_sam_1 = df_all_sam_1['dice_all'].min()
    dice_score_minimum_sam_2 = df_all_sam_2['dice_all'].min()
    dice_score_minimum_delta = dice_score_minimum_sam_2 - dice_score_minimum_sam_1

    dice_score_maximum_sam_1 = df_all_sam_1['dice_all'].max()
    dice_score_maximum_sam_2 = df_all_sam_2['dice_all'].max()
    dice_score_maximum_delta = dice_score_maximum_sam_2 - dice_score_maximum_sam_1

    dice_score_average_sam_1 = df_all_sam_1['dice_all'].mean()
    dice_score_average_sam_2 = df_all_sam_2['dice_all'].mean()
    dice_score_average_delta = dice_score_average_sam_2 - dice_score_average_sam_1

    dice_score_standard_deviation_sam_1 = df_all_sam_1['dice_all'].std()
    dice_score_standard_deviation_sam_2 = df_all_sam_2['dice_all'].std()
    dice_score_standard_deviation_delta = dice_score_standard_deviation_sam_2 - dice_score_standard_deviation_sam_1

    sam_score_minimum_sam_1 = df_all_sam_1['sam_score_all'].min()
    sam_score_minimum_sam_2 = df_all_sam_2['sam_score_all'].min()
    sam_score_minimum_delta = sam_score_minimum_sam_2 - sam_score_minimum_sam_1

    sam_score_maximum_sam_1 = df_all_sam_1['sam_score_all'].max()
    sam_score_maximum_sam_2 = df_all_sam_2['sam_score_all'].max()
    sam_score_maximum_delta = sam_score_maximum_sam_2 - sam_score_maximum_sam_1

    sam_score_average_sam_1 = df_all_sam_1['sam_score_all'].mean()
    sam_score_average_sam_2 = df_all_sam_2['sam_score_all'].mean()
    sam_score_average_delta = sam_score_average_sam_2 - sam_score_average_sam_1

    sam_score_standard_deviation_sam_1 = df_all_sam_1['sam_score_all'].std()
    sam_score_standard_deviation_sam_2 = df_all_sam_2['sam_score_all'].std()
    sam_score_standard_deviation_delta = sam_score_standard_deviation_sam_2 - sam_score_standard_deviation_sam_1

    dictionary = {
        'columns': [
            'jaccard_index_sam_1', 'jaccard_index_sam_2', 'jaccard_index_delta',
            'dice_score_sam_1', 'dice_score_sam_2', 'dice_score_delta',
            'sam_score_sam_1', 'sam_score_sam_2', 'sam_score_delta'],
        'minimum': [
            jaccard_index_minimum_sam_1, jaccard_index_minimum_sam_2, jaccard_index_minimum_delta,
            dice_score_minimum_sam_1, dice_score_minimum_sam_2, dice_score_minimum_delta,
            sam_score_minimum_sam_1, sam_score_minimum_sam_2, sam_score_minimum_delta],
        'maximum': [
            jaccard_index_maximum_sam_1, jaccard_index_maximum_sam_2, jaccard_index_maximum_delta,
            dice_score_maximum_sam_1, dice_score_maximum_sam_2, dice_score_maximum_delta,
            sam_score_maximum_sam_1, sam_score_maximum_sam_2, sam_score_maximum_delta],
        'average': [
            jaccard_index_average_sam_1, jaccard_index_average_sam_2, jaccard_index_average_delta,
            dice_score_average_sam_1, dice_score_average_sam_2, dice_score_average_delta,
            sam_score_average_sam_1, sam_score_average_sam_2, sam_score_average_delta],
        'standard_deviation': [
            jaccard_index_standard_deviation_sam_1, jaccard_index_standard_deviation_sam_2,
            jaccard_index_standard_deviation_delta,
            dice_score_standard_deviation_sam_1, dice_score_standard_deviation_sam_2,
            dice_score_standard_deviation_delta,
            sam_score_standard_deviation_sam_1, sam_score_standard_deviation_sam_2, sam_score_standard_deviation_delta],
    }
    df = pd.DataFrame.from_dict(dictionary)
    df = df.transpose()
    return df


def create_comparison_boxplot(paper_version: bool = False):
    df_coronacases_sam_1 = load_data(Path('../working_data/covid/results/sam1/2-coronacases/'))
    df_coronacases_sam_1.describe()
    df_coronacases_sam_2 = load_data(Path('../working_data/covid/results/sam2/2-coronacases/'))
    df_coronacases_sam_2.describe()
    df_radiopaedia_sam_1 = load_data(Path('../working_data/covid/results/sam1/3-radiopaedia/'))
    df_radiopaedia_sam_1.describe()
    df_radiopaedia_sam_2 = load_data(Path('../working_data/covid/results/sam2/3-radiopaedia/'))
    df_radiopaedia_sam_2.describe()
    df_montgomery_sam_1 = load_data(Path('../working_data/montgomery/results/sam1_all/'))
    df_montgomery_sam_1.describe()
    df_montgomery_sam_2 = load_data(Path('../working_data/montgomery/results/sam2_all/'))
    df_montgomery_sam_2.describe()

    coronacases_sam_1_vs_sam_2_df = get_sam_1_vs_sam_2_dataframe(df_coronacases_sam_1, df_coronacases_sam_2)
    radiopaedia_sam_1_vs_sam_2_df = get_sam_1_vs_sam_2_dataframe(df_radiopaedia_sam_1, df_radiopaedia_sam_2)
    montgomery_sam_1_vs_sam_2_df = get_sam_1_vs_sam_2_dataframe(df_montgomery_sam_1, df_montgomery_sam_2)

    plt.rcParams.update(PyPlotParameters)
    figure_size = (
        PYPLOT_PLOT_WIDTH / PYPLOT_PLOT_DPI,
        PYPLOT_PLOT_HEIGHT / PYPLOT_PLOT_DPI)

    fig = plt.figure(figsize=figure_size, dpi=PYPLOT_PLOT_DPI)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=False))
    ax.set_ylim(0.70, 1.0)
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.grid(linewidth=1, which='major')
    ax.grid(linewidth=0.5, which='minor')

    df_coronacases_sam_1.boxplot(
        column=['sam_score_all'],
        positions=[0],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lavender'},
        ax=ax)
    df_coronacases_sam_2.boxplot(
        column=['sam_score_all'],
        positions=[1],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lavender'},
        ax=ax)
    df_radiopaedia_sam_1.boxplot(
        column=['sam_score_all'],
        positions=[2],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lightblue'},
        ax=ax)
    df_radiopaedia_sam_2.boxplot(
        column=['sam_score_all'],
        positions=[3],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lightblue'},
        ax=ax)
    df_montgomery_sam_1.boxplot(
        column=['sam_score_all'],
        positions=[4],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'bisque'},
        ax=ax)
    df_montgomery_sam_2.boxplot(
        column=['sam_score_all'],
        positions=[5],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'bisque'},
        ax=ax)

    ax.set_xticks(
        list(range(6)),
        ['SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2'])
    secondary_xaxis = ax.secondary_xaxis(location=-0.075)
    secondary_xaxis.tick_params(bottom=False)
    secondary_xaxis.set_xticks(
        [0.5, 2.5, 4.5],
        labels=['Coronacases', 'Radiopaedia', 'Montgomery'])

    if not paper_version:
        ax.set_title('SAM Score Comparison')

    # ax.set_xticks([0, 1], ['Jaccard', 'Dice'])
    plt.show()

    fig.savefig('sam2_delta_report/boxplot_sam_score.pdf', bbox_inches='tight')
    fig.savefig('sam2_delta_report/boxplot_sam_score.png', bbox_inches='tight')


def main():
    paper_version = True
    create_comparison_boxplot(paper_version=paper_version)


if __name__ == '__main__':
    main()
