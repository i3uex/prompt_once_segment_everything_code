import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
from rich import print
from scipy.stats import ttest_rel

from data_model.dataset import Dataset
from data_model.metric import Metric
from plots import common, constants


def hypothesis_test(dataset: Dataset, metric: Metric):
    # Load the datasets
    if dataset is Dataset.Coronacases:
        file1 = '../working_data/covid/results/sam1/2-coronacases/joint_raw_data_2024-08-07_11-33-32.csv'
        file2 = '../working_data/covid/results/sam2/2-coronacases/joint_raw_data_2024-08-07_13-05-37.csv'
    elif dataset is Dataset.Radiopaedia:
        file1 = '../working_data/covid/results/sam1/3-radiopaedia/joint_raw_data_2024-08-07_11-34-07.csv'
        file2 = '../working_data/covid/results/sam2/3-radiopaedia/joint_raw_data_2024-08-07_13-05-49.csv'
    elif dataset is Dataset.Montgomery:
        file1 = '../working_data/montgomery/results/sam1_all/joint_raw_data_2024-08-08_10-25-06.csv'
        file2 = '../working_data/montgomery/results/sam2_all/joint_raw_data_2024-08-08_10-36-28.csv'
    else:
        raise ValueError(f'Invalid dataset: {dataset.description}')

    # Read the data
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Extract the 'jaccard' column from both datasets
    column1 = data1[metric.column_name]
    column2 = data2[metric.column_name]

    # Remove rows with missing values in the 'jaccard' column
    column1_clean = column1.dropna()
    column2_clean = column2.dropna()

    # Ensure that after dropping NaNs, both datasets are still aligned and of the same length
    column1_clean = column1_clean.reset_index(drop=True)
    column2_clean = column2_clean.reset_index(drop=True)

    # Perform the paired t-test
    t_statistic, p_value = ttest_rel(column1_clean, column2_clean)

    # Output the results
    print(f'\nHypothesis test for {dataset.description} and {metric.description}')
    print('Hypothesis: There is no significant difference between SAM and SAM 2.')
    print(f'T-statistic: {t_statistic}, P-value: {p_value}')

    # Interpretation
    if p_value < 0.05:
        result = 'Reject'
        print('Reject the null hypothesis: There is a significant difference between the two systems')
        if t_statistic > 0:
            interpretation = 'SAM is better than SAM 2'
        else:
            interpretation = 'SAM 2 is better than SAM'
    else:
        result = 'Fail to reject'
        interpretation = 'No significant difference between the two systems'

    print(result)
    print(interpretation)

    data = {
        'dataset': dataset.description,
        'metric': metric.description,
        't_statistic': t_statistic,
        'p_value': p_value,
        'result': result,
        'interpretation': interpretation
    }
    return data


def main():
    datasets = [Dataset.Coronacases, Dataset.Radiopaedia, Dataset.Montgomery]
    metrics = [Metric.JaccardIndex, Metric.DiceScore]

    results = []
    for dataset in datasets:
        for metric in metrics:
            result = hypothesis_test(
                dataset=dataset,
                metric=metric
            )
            results.append(result)

    df = pd.DataFrame(results)
    df.to_csv('sam2_delta_report/hypothesis_test_results.csv', index=False)

    # Set up PyPlot

    # Create a DataFrame to organize the results
    data = {
        'Dataset': df['dataset'],
        'Metric': df['metric'],
        'T-statistic': df['t_statistic'],
        'P-value': df['p_value'],
        'result': df['result']
    }

    df = pd.DataFrame(data)

    # Set up the plots
    common.setup_pyplot()
    plt.rcParams.update({'axes.titlesize': 20})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'xtick.labelsize': 20})
    plt.rcParams.update({'ytick.labelsize': 20})

    figure_size = constants.PYPLOT_FIGURE_SIZE_DOUBLE_HEIGHT
    fig, ax = plt.subplots(2, 1, figsize=figure_size, dpi=constants.PYPLOT_PLOT_DPI)
    fig.subplots_adjust(hspace=-1)
    for axis in ax:
        # axis.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=False))
        # axis.set_ylim(-0.25, 0.79)
        axis.yaxis.set_minor_locator(mtick.AutoMinorLocator())
        axis.grid(linewidth=1, which='major')
        axis.grid(linewidth=0.5, which='minor')

    # Bar plot for T-statistics
    ax[0].bar(df['Dataset'] + ' - ' + df['Metric'], df['T-statistic'],
              color=['slateblue' if 'Fail to reject' in i else 'orangered' for i in df['result']], zorder=10)
    ax[0].axhline(0, color='black', linewidth=0.8)
    ax[0].set_title('T-statistics of Paired T-Tests')
    ax[0].set_ylabel('T-statistic')
    ax[0].set_xticklabels(df['Metric'])

    secondary_xaxis = ax[0].secondary_xaxis(location=-0.050)
    secondary_xaxis.tick_params(bottom=False)
    # secondary_xaxis.set_xticks(
    #     [0.5, 2.5, 4.5],
    #     labels=['\nJaccard Index', '\nDice Score', '\nSAM Score'])
    secondary_xaxis.set_xticks(
        [0.5, 2.5, 4.5],
        labels=['Coronacases', 'Radiopaedia', 'Montgomery'])

    # Bar plot for P-values
    ax[1].bar(df['Dataset'] + ' - ' + df['Metric'], df['P-value'],
              color=['slateblue' if 'Fail to reject' in i else 'orangered' for i in df['result']], zorder=10)
    ax[1].axhline(0.05, color='seagreen', linestyle='--', linewidth=1, label='Significance Threshold (0.05)', zorder=11)
    ax[1].set_title('P-values of Paired T-Tests')
    ax[1].set_ylabel('P-value')
    ax[1].set_xticklabels(df['Metric'])
    ax[1].set_yscale('log')
    ax[1].legend()

    secondary_xaxis = ax[1].secondary_xaxis(location=-0.050)
    secondary_xaxis.tick_params(bottom=False)
    # secondary_xaxis.set_xticks(
    #     [0.5, 2.5, 4.5],
    #     labels=['\nJaccard Index', '\nDice Score', '\nSAM Score'])
    secondary_xaxis.set_xticks(
        [0.5, 2.5, 4.5],
        labels=['Coronacases', 'Radiopaedia', 'Montgomery'])


    plt.tight_layout()
    plt.show()

    fig.savefig('sam2_delta_report/hypothesis_test_results.pdf', bbox_inches='tight')
    fig.savefig('sam2_delta_report/hypothesis_test_results.png', bbox_inches='tight')


if __name__ == '__main__':
    main()
