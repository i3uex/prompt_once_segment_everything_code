from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from data_model.ReportType import ReportType
from data_model.dataset import Dataset
from tools.files import get_most_recent_timestamped_file

PYPLOT_PLOT_WIDTH = 1280
PYPLOT_PLOT_HEIGHT = 960
PYPLOT_PLOT_DPI = 100
PYPLOT_SIZE = 30
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


def load_data(results_folder_path: Path) -> (pd.DataFrame, pd.DataFrame):
    file_path = get_most_recent_timestamped_file(
        results_folder_path, RawDataFilePattern)
    df_all = pd.read_csv(file_path)

    df_segmentation_failures = df_all[(df_all['jaccard'] == 0) | (df_all['dice'] == 0)]
    if df_segmentation_failures.size == 0:
        df_segmentation_failures = None
    else:
        df_all.drop(df_segmentation_failures.index, inplace=True)

    df_all.rename(columns={
        'jaccard': 'jaccard_all',
        'dice': 'dice_all',
        'sam_score': 'sam_score_all'
    }, inplace=True)

    return df_all, df_segmentation_failures


def get_sam1_vs_sam2_dataframe(df_all_sam_1: pd.DataFrame, df_all_sam_2: pd.DataFrame) -> pd.DataFrame:
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


def add_sam1_vs_sam2_deltas(
        sam1_vs_sam2_df: pd.DataFrame
):
    jaccard_mean_sam1 = sam1_vs_sam2_df[0]['average']
    jaccard_mean_sam2 = sam1_vs_sam2_df[1]['average']
    jaccard_mean_delta = sam1_vs_sam2_df[2]['average']

    plt.vlines(
        x=0 + 0.5,
        ymin=jaccard_mean_sam1,
        ymax=jaccard_mean_sam2,
        color='darkgreen', linestyle=(0, (3, 1)), linewidth=2
    )
    if jaccard_mean_delta > 0:
        y_text = jaccard_mean_sam2
        plot_text = f'+{jaccard_mean_delta * 100:.2f} %'
    else:
        y_text = jaccard_mean_sam1
        plot_text = f'{jaccard_mean_delta * 100:.2f} %'
    plt.text(0 + 0.5, y_text + 0.0025, plot_text, fontsize=10,
             color='darkgreen',
             ha='center', va='bottom')

    dice_mean_sam1 = sam1_vs_sam2_df[3]['average']
    dice_mean_sam2 = sam1_vs_sam2_df[4]['average']
    dice_mean_delta = sam1_vs_sam2_df[5]['average']

    plt.vlines(
        x=2 + 0.5,
        ymin=dice_mean_sam1,
        ymax=dice_mean_sam2,
        color='darkgreen', linestyle=(0, (3, 1)), linewidth=2
    )
    if dice_mean_delta > 0:
        y_text = dice_mean_sam2
        plot_text = f'+{dice_mean_delta * 100:.2f} %'
    else:
        y_text = dice_mean_sam1
        plot_text = f'{dice_mean_delta * 100:.2f} %'
    plt.text(2 + 0.5, y_text + 0.0025, plot_text, fontsize=10,
             color='darkgreen',
             ha='center', va='bottom')


def add_sota_reference(
        dataset: Dataset,
        sam1_vs_sam2_df: pd.DataFrame
):
    # Add SotA averages to compare. Coronacases and Radiopaedia contain CTs,
    # Montgomery contains X-rays.
    # Owais et al.
    sota_ct_jaccard = 0.9738
    sota_ct_dice = 0.9866
    # Chen et al.
    sota_xray_jaccard = 0.9782
    sota_xray_dice = 0.9888

    if dataset is Dataset.Coronacases or dataset is Dataset.Radiopaedia:
        sota_jaccard = sota_ct_jaccard
        sota_dice = sota_ct_dice
    elif dataset is Dataset.Montgomery:
        sota_jaccard = sota_xray_jaccard
        sota_dice = sota_xray_dice
    else:
        raise ValueError(f'Invalid dataset: {dataset.description}')

    plt.hlines(
        xmin=0 - 0.2,
        xmax=1 + 0.2,
        y=sota_jaccard,
        color='royalblue', linestyle='solid', linewidth=2,
        label=f'{sota_jaccard * 100:.2f} % (Owais et al.)',
        zorder=5,
        path_effects=[pe.SimpleLineShadow(shadow_color='grey'), pe.Normal()]
    )

    jaccard_mean_sam2 = sam1_vs_sam2_df[1]['average']
    jaccard_mean_delta = sota_jaccard - jaccard_mean_sam2
    plt.vlines(
        x=0 + 1.30,
        ymin=jaccard_mean_sam2,
        ymax=sota_jaccard,
        color='indianred', linestyle=(0, (3, 1)), linewidth=2
    )
    plt.text(0 + 1.30, jaccard_mean_sam2 - 0.012, f'-{jaccard_mean_delta * 100:.2f} %', fontsize=10,
             color='indianred',
             ha='center', va='bottom')

    plt.hlines(
        xmin=2 - 0.2,
        xmax=3 + 0.2,
        y=sota_dice,
        color='orange', linestyle='solid', linewidth=2,
        label=f'{sota_dice * 100:.2f} % (Chen et al.)',
        zorder=5,
        path_effects=[pe.SimpleLineShadow(shadow_color='grey'), pe.Normal()]
    )

    dice_mean_sam2 = sam1_vs_sam2_df[4]['average']
    dice_mean_delta = sota_dice - dice_mean_sam2
    plt.vlines(
        x=2 + 1.30,
        ymin=dice_mean_sam2,
        ymax=sota_dice,
        color='indianred', linestyle=(0, (3, 1)), linewidth=2
    )
    plt.text(2 + 1.30, dice_mean_sam2 - 0.012, f'-{dice_mean_delta * 100:.2f} %', fontsize=10,
             color='indianred',
             ha='center', va='bottom')


def create_comparison_boxplot(dataset: Dataset, report_type: ReportType, paper_version: bool = False):
    if report_type == ReportType.Image:
        if dataset is Dataset.Coronacases:
            df_all_sam1, _ = load_data(Path('working_data/covid/results/sam1/2-coronacases/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/covid/results/sam2/2-coronacases/'))
            df_all_sam2.describe()
        elif dataset is Dataset.Radiopaedia:
            df_all_sam1, _ = load_data(Path('working_data/covid/results/sam1/3-radiopaedia/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/covid/results/sam2/3-radiopaedia/'))
            df_all_sam2.describe()
        elif dataset is Dataset.Montgomery:
            df_all_sam1, _ = load_data(Path('working_data/montgomery/results/sam1_all/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/montgomery/results/sam2_all/'))
            df_all_sam2.describe()
        else:
            raise ValueError(f'Invalid dataset: {dataset.description}')
    elif report_type == ReportType.Video:
        if dataset is Dataset.Coronacases:
            df_all_sam1, _ = load_data(Path('working_data/covid/results/sam1/2-coronacases/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/covid/sam2_videos/results/2-coronacases/'))
            df_all_sam2.describe()
        elif dataset is Dataset.Radiopaedia:
            df_all_sam1, _ = load_data(Path('working_data/covid/results/sam1/3-radiopaedia/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/covid/sam2_videos/results/3-radiopaedia/'))
            df_all_sam2.describe()
        else:
            raise ValueError(f'Invalid dataset: {dataset.description}')
    elif report_type == ReportType.VideoFakeSeed:
        if dataset is Dataset.Coronacases:
            df_all_sam1, _ = load_data(Path('working_data/covid/sam2_videos/results/2-coronacases/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/covid/sam2_videos_fake_seed/results/2-coronacases/'))
            df_all_sam2.describe()
        elif dataset is Dataset.Radiopaedia:
            df_all_sam1, _ = load_data(Path('working_data/covid/sam2_videos/results/3-radiopaedia/'))
            df_all_sam1.describe()
            df_all_sam2, df_sam2_segmentation_failures = load_data(Path('working_data/covid/sam2_videos_fake_seed/results/3-radiopaedia/'))
            df_all_sam2.describe()
        else:
            raise ValueError(f'Invalid dataset: {dataset.description}')
    else:
        raise ValueError(f'Invalid report type: {report_type}')

    sam1_vs_sam2_df = get_sam1_vs_sam2_dataframe(df_all_sam1, df_all_sam2)

    plt.rcParams.update(PyPlotParameters)
    figure_size = (
        PYPLOT_PLOT_WIDTH / PYPLOT_PLOT_DPI,
        PYPLOT_PLOT_HEIGHT / PYPLOT_PLOT_DPI)

    fig = plt.figure(figsize=figure_size, dpi=PYPLOT_PLOT_DPI)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=False))
    if report_type is ReportType.Image:
        ax.set_ylim(0.70, 1.0)
    elif report_type is ReportType.Video:
        ax.set_ylim(0.82, 1.0)
    elif report_type is ReportType.VideoFakeSeed:
        ax.set_ylim(0.845, 1.005)
    else:
        raise ValueError(f'Invalid report type: {report_type}')
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.grid(linewidth=1, which='major')
    ax.grid(linewidth=0.5, which='minor')

    df_all_sam1.boxplot(
        column=['jaccard_all'],
        positions=[0],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lavender'},
        ax=ax)
    df_all_sam2.boxplot(
        column=['jaccard_all'],
        positions=[1],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lavender'},
        ax=ax)
    df_all_sam1.boxplot(
        column=['dice_all'],
        positions=[2],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lightblue'},
        ax=ax)
    df_all_sam2.boxplot(
        column=['dice_all'],
        positions=[3],
        widths=0.3,
        showfliers=False,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        notch=True,
        boxprops={'facecolor': 'lightblue'},
        ax=ax)
    # df_all_sam_1.boxplot(
    #     column=['sam_score_all'],
    #     positions=[4],
    #     showfliers=False,
    #     patch_artist=True,
    #     showmeans=True,
    #     meanline=True,
    #     notch=True,
    #     boxprops={'facecolor': 'bisque'},
    #     ax=ax)
    # df_all_sam_2.boxplot(
    #     column=['sam_score_all'],
    #     positions=[5],
    #     showfliers=False,
    #     patch_artist=True,
    #     showmeans=True,
    #     meanline=True,
    #     notch=True,
    #     boxprops={'facecolor': 'bisque'},
    #     ax=ax)

    add_sam1_vs_sam2_deltas(
        sam1_vs_sam2_df=sam1_vs_sam2_df
    )
    add_sota_reference(
        dataset=dataset,
        sam1_vs_sam2_df=sam1_vs_sam2_df
    )

    # ax.set_xticks(
    #     list(range(6)),
    #     ['SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2'])
    if report_type is ReportType.Image or report_type is ReportType.Video:
        ax.set_xticks(
            list(range(4)),
            ['SAM', 'SAM 2', 'SAM', 'SAM 2'])
    elif report_type is ReportType.VideoFakeSeed:
        ax.set_xticks(
            list(range(4)),
            ['Middle', 'Shared', 'Middle', 'Shared'])
    secondary_xaxis = ax.secondary_xaxis(location=-0.075)
    secondary_xaxis.tick_params(bottom=False)
    # secondary_xaxis.set_xticks(
    #     [0.5, 2.5, 4.5],
    #     labels=['\nJaccard Index', '\nDice Score', '\nSAM Score'])
    secondary_xaxis.set_xticks(
        [0.5, 2.5],
        labels=['Jaccard Index', 'Dice Score'])

    plt.ylabel('Metric Value')
    if paper_version:
        if dataset is Dataset.Radiopaedia or dataset is Dataset.Montgomery:
            plt.tick_params(labelleft=False)

    plt.legend(loc='lower right')

    if not paper_version:
        if dataset is Dataset.Coronacases:
            ax.set_title('Lung CT (Coronacases Dataset)')
        elif dataset is Dataset.Radiopaedia:
            ax.set_title('Lung CBCT (Radiopaedia Dataset)')
        elif dataset is Dataset.Montgomery:
            ax.set_title('Sagital Lung X-rays (Montgomery Dataset)')
        else:
            raise ValueError(f'Invalid dataset: {dataset.description}')

    # ax.set_xticks([0, 1], ['Jaccard', 'Dice'])
    plt.show()

    if report_type is ReportType.Image:
        report_base_path = Path('comparison/sam2_delta_report_image/')
    elif report_type is ReportType.Video:
        report_base_path = Path('comparison/sam2_delta_report_video/')
    elif report_type is ReportType.VideoFakeSeed:
        report_base_path = Path('comparison/sam2_delta_report_video_fake_seed/')
    else:
        raise ValueError(f'Invalid report type: {report_type}')

    if dataset is Dataset.Coronacases:
        coronacases_report_folder = report_base_path / 'coronacases/'
        coronacases_report_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_base_path / 'coronacases/boxplot_coronacases.pdf', bbox_inches='tight')
        fig.savefig(report_base_path / 'coronacases/boxplot_coronacases.png', bbox_inches='tight')
        sam1_vs_sam2_df.to_csv(report_base_path / 'coronacases/boxplot_coronacases.csv')
        if report_type is ReportType.Video:
            df_sam2_segmentation_failures.to_csv(report_base_path / 'coronacases/coronacases_video_failures.csv')
    elif dataset is Dataset.Radiopaedia:
        radiopaedia_report_folder = report_base_path / 'radiopaedia/'
        radiopaedia_report_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_base_path / 'radiopaedia/boxplot_radiopaedia.pdf', bbox_inches='tight')
        fig.savefig(report_base_path / 'radiopaedia/boxplot_radiopaedia.png', bbox_inches='tight')
        sam1_vs_sam2_df.to_csv(report_base_path / 'radiopaedia/boxplot_radiopaedia.csv')
        if report_type is ReportType.Video:
            df_sam2_segmentation_failures.to_csv(report_base_path / 'radiopaedia/radiopaedia_video_failures.csv')
    elif dataset is Dataset.Montgomery:
        montgomery_report_folder = report_base_path / 'montgomery/'
        montgomery_report_folder.mkdir(parents=True, exist_ok=True)
        fig.savefig(report_base_path / 'montgomery/boxplot_montgomery.pdf', bbox_inches='tight')
        fig.savefig(report_base_path / 'montgomery/boxplot_montgomery.png', bbox_inches='tight')
        sam1_vs_sam2_df.to_csv(report_base_path / 'montgomery/boxplot_montgomery.csv')
    else:
        raise ValueError(f'Invalid dataset: {dataset.description}')


def main():
    report_type = ReportType.VideoFakeSeed

    if report_type is ReportType.Image:
        datasets = [Dataset.Coronacases, Dataset.Radiopaedia, Dataset.Montgomery]
    elif report_type is ReportType.Video or report_type is ReportType.VideoFakeSeed:
        datasets = [Dataset.Coronacases, Dataset.Radiopaedia]
    else:
        raise ValueError(f'Invalid report type: {report_type}')

    paper_version = True

    for dataset in datasets:
        create_comparison_boxplot(
            dataset=dataset,
            report_type=report_type,
            paper_version=paper_version)


if __name__ == '__main__':
    main()
