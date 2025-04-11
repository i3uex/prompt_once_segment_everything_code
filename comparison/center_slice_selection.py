import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd

from data_model.dataset import Dataset
from data_model.metric import Metric
from plots import common, constants


def create_center_slice_selection_plot(
        dataset: Dataset,
        metric: Metric,
        paper_version: bool
):
    # Load the CSV file
    if dataset is Dataset.Coronacases:
        file_path = 'working_data/covid/results/sam1/2-coronacases/joint_raw_data_2024-08-07_11-33-32.csv'
    elif dataset is Dataset.Radiopaedia:
        file_path = 'working_data/covid/results/sam1/3-radiopaedia/joint_raw_data_2024-08-07_11-34-07.csv'
    else:
        raise ValueError(f'Invalid dataset: {dataset}')

    data = pd.read_csv(file_path)

    # Fill NaN values with 0
    filtered_data = data
    filtered_data[metric.column_name] = filtered_data[metric.column_name].fillna(0)

    # Normalizing the slice numbers (scaling between 0 and 1)
    filtered_data['normalized_slice'] = (
            (filtered_data['slice'] - filtered_data['slice'].min()) /
            (filtered_data['slice'].max() - filtered_data['slice'].min()))

    # Recalculate the average metric values using the normalized slice numbers
    average_metric_normalized = filtered_data.groupby('normalized_slice')[metric.column_name].mean().reset_index()

    # Calculating the global average metric value
    global_average_metric = filtered_data[metric.column_name].mean()

    # Calculating the Xth percentile for the metric values
    top_1 = 0.01
    top_2 = 0.02
    metric_1th = filtered_data[metric.column_name].quantile(1 - top_1)
    metric_2th = filtered_data[metric.column_name].quantile(1 - top_2)

    # Identifying the slice range where the top X% of metric values are located
    top_1_normalized_slices = filtered_data[filtered_data[metric.column_name] >= metric_1th]['normalized_slice']
    top_2_normalized_slices = filtered_data[filtered_data[metric.column_name] >= metric_2th]['normalized_slice']

    # Determine the range (min and max slice numbers)
    min_1_normalized_slice = top_1_normalized_slices.min()
    max_1_normalized_slice = top_1_normalized_slices.max()
    min_2_normalized_slice = top_2_normalized_slices.min()
    max_2_normalized_slice = top_2_normalized_slices.max()

    # Set the plot up
    common.setup_pyplot()
    figure_size = constants.PYPLOT_FIGURE_SIZE
    figure, axes = plt.subplots(figsize=figure_size, dpi=constants.PYPLOT_PLOT_DPI)
    axes.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=False))
    axes.xaxis.set_minor_locator(mtick.AutoMinorLocator())
    axes.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=False))
    axes.yaxis.set_minor_locator(mtick.AutoMinorLocator())

    plt.ylim(0.74, 1.025)

    # Plotting the data with the normalized slice numbers and top 1% slice range highlighted, including a vertical line at 0.5
    plt.scatter(average_metric_normalized['normalized_slice'], average_metric_normalized[metric.code], color='green',
                alpha=0.5, label=f'{metric.description} per slice position', zorder=20)
    # plt.axhline(y=global_average_metric, color='red', linestyle='--',
    #             label=f'Average {metric.description}: {global_average_metric * 100:.2f} %', zorder=10)
    plt.axhline(y=global_average_metric, color='red', linestyle='--',
                label=f'Average {metric.description}', zorder=10)

    # Highlighting the range where the top X% metric values are found (normalized)
    # plt.axvspan(min_normalized_slice, max_normalized_slice, color='yellow', alpha=0.3,
    #             label=f'Slice position range ({min_normalized_slice * 100:.2f} % - {max_normalized_slice * 100:.2f} %) with Top {int(top * 100)} % {metric.description}')
    plt.axvspan(min_1_normalized_slice, max_1_normalized_slice, color='orange', alpha=0.3,
                label=f'Top {int(top_1 * 100)} % {metric.description}')
    plt.axvspan(min_2_normalized_slice, max_2_normalized_slice, color='yellow', alpha=0.3,
                label=f'Top {int(top_2 * 100)} % {metric.description}')

    # Adding a vertical line at 0.5
    plt.axvline(x=0.5, color='blue', linestyle='-', label='Center slice position', zorder=10)

    if not paper_version:
        plt.title(
            f'Average {metric.description} vs. Normalized Slice Number with Top {int(top * 100)} % {metric.description} Slice Range')
    plt.xlabel('Slice Position')
    plt.ylabel(f'{metric.description}')
    plt.legend().set_zorder(100)
    plt.grid(True)
    plt.show()

    figure.savefig(f'comparison/center_slice/{dataset.code}_{metric.code}.pdf', bbox_inches='tight')
    figure.savefig(f'comparison/center_slice/{dataset.code}_{metric.code}.png', bbox_inches='tight')


def main():
    datasets = [Dataset.Coronacases, Dataset.Radiopaedia]
    metrics = [Metric.JaccardIndex, Metric.DiceScore]
    # datasets = [Dataset.Coronacases]
    # metrics = [Metric.JaccardIndex]

    paper_version = True

    for dataset in datasets:
        for metric in metrics:
            create_center_slice_selection_plot(
                dataset=dataset,
                metric=metric,
                paper_version=paper_version
            )


if __name__ == '__main__':
    main()
