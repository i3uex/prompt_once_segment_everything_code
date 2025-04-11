import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from csv_keys import SetImageTimeKey, PredictTimeKey
from plots import common, constants

paper_version = True

# Load the datasets
file_paths = {
    'coronacases_sam1': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/covid/results/sam1/2-coronacases/joint_raw_data_2024-08-07_11-33-32.csv',
    'radiopaedia_sam1': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/covid/results/sam1/3-radiopaedia/joint_raw_data_2024-08-07_11-34-07.csv',
    'montgomery_sam1_all': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam1_all/joint_raw_data_2024-08-08_10-25-06.csv',
    'montgomery_sam1_1by1': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam1_1by1/joint_raw_data_2024-08-07_16-51-27.csv',
    'montgomery_sam1_250': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam1_250/joint_raw_data_2024-08-07_21-24-34.csv',
    'coronacases_sam2': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/covid/results/sam2/2-coronacases/joint_raw_data_2024-08-07_13-05-37.csv',
    'radiopaedia_sam2': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/covid/results/sam2/3-radiopaedia/joint_raw_data_2024-08-07_13-05-49.csv',
    'montgomery_sam2_all': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam2_all/joint_raw_data_2024-08-08_10-36-28.csv',
    'montgomery_sam2_1by1': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam2_1by1/joint_raw_data_2024-08-07_20-39-18.csv',
    'montgomery_sam2_250': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam2_250/joint_raw_data_2024-08-07_21-40-30.csv',
}

datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Prepare data for each dataset
coronacases_sam1 = datasets['coronacases_sam1'][[SetImageTimeKey, PredictTimeKey]].dropna()
coronacases_sam2 = datasets['coronacases_sam2'][[SetImageTimeKey, PredictTimeKey]].dropna()
radiopaedia_sam1 = datasets['radiopaedia_sam1'][[SetImageTimeKey, PredictTimeKey]].dropna()
radiopaedia_sam2 = datasets['radiopaedia_sam2'][[SetImageTimeKey, PredictTimeKey]].dropna()
montgomery_sam1_all = datasets['montgomery_sam1_all'][[SetImageTimeKey, PredictTimeKey]].dropna()
montgomery_sam2_all = datasets['montgomery_sam2_all'][[SetImageTimeKey, PredictTimeKey]].dropna()
montgomery_sam1_1by1 = datasets['montgomery_sam1_1by1'][[SetImageTimeKey, PredictTimeKey]].dropna()
montgomery_sam2_1by1 = datasets['montgomery_sam2_1by1'][[SetImageTimeKey, PredictTimeKey]].dropna()
montgomery_sam1_250 = datasets['montgomery_sam1_250'][[SetImageTimeKey, PredictTimeKey]].dropna()
montgomery_sam2_250 = datasets['montgomery_sam2_250'][[SetImageTimeKey, PredictTimeKey]].dropna()

# Calculate the average set_image_time and predict_time for all datasets
coronacases_sam1_avg = coronacases_sam1.mean()
coronacases_sam2_avg = coronacases_sam2.mean()
radiopaedia_sam1_avg = radiopaedia_sam1.mean()
radiopaedia_sam2_avg = radiopaedia_sam2.mean()
montgomery_sam1_all_avg = montgomery_sam1_all.mean()
montgomery_sam2_all_avg = montgomery_sam2_all.mean()
montgomery_sam1_1by1_avg = montgomery_sam1_1by1.mean()
montgomery_sam2_1by1_avg = montgomery_sam2_1by1.mean()
montgomery_sam1_250_avg = montgomery_sam1_250.mean()
montgomery_sam2_250_avg = montgomery_sam2_250.mean()

# Create a stacked bar plot for the average times of SAM1 and SAM2 for all datasets with the final requested adjustments
common.setup_pyplot()
figure_size = constants.PYPLOT_FIGURE_SIZE
figure, axes = plt.subplots(figsize=figure_size, dpi=constants.PYPLOT_PLOT_DPI)
# axes.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1, decimals=False))
axes.yaxis.set_minor_locator(mtick.AutoMinorLocator())
axes.grid(linewidth=1, which='major')
axes.grid(linewidth=0.5, which='minor')

bar_width = 0.50
# index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Bars for Coronacases SAM1, SAM2, Radiopaedia SAM1, SAM2, Montgomery SAM1, SAM2
index = [0, 1, 2, 3, 4, 5]  # Bars for Coronacases SAM1, SAM2, Radiopaedia SAM1, SAM2, Montgomery SAM1, SAM2

# Colors for consistency
color_set_image_time = 'orange'
color_predict_time = 'red'

# Coronacases SAM1
bar1 = axes.bar(index[0], coronacases_sam1_avg[SetImageTimeKey], bar_width, color=color_set_image_time, label='Set Image Time', zorder=10)
bar2 = axes.bar(index[0], coronacases_sam1_avg[PredictTimeKey], bar_width, bottom=coronacases_sam1_avg[SetImageTimeKey], color=color_predict_time, label='Predict Time', zorder=10)

# Coronacases SAM2
bar3 = axes.bar(index[1], coronacases_sam2_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
bar4 = axes.bar(index[1], coronacases_sam2_avg[PredictTimeKey], bar_width, bottom=coronacases_sam2_avg[SetImageTimeKey], color=color_predict_time, zorder=10)

# Radiopaedia SAM1
bar5 = axes.bar(index[2], radiopaedia_sam1_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
bar6 = axes.bar(index[2], radiopaedia_sam1_avg[PredictTimeKey], bar_width, bottom=radiopaedia_sam1_avg[SetImageTimeKey], color=color_predict_time, zorder=10)

# Radiopaedia SAM2
bar7 = axes.bar(index[3], radiopaedia_sam2_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
bar8 = axes.bar(index[3], radiopaedia_sam2_avg[PredictTimeKey], bar_width, bottom=radiopaedia_sam2_avg[SetImageTimeKey], color=color_predict_time, zorder=10)

# Montgomery SAM1 all
bar9 = axes.bar(index[4], montgomery_sam1_all_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
bar10 = axes.bar(index[4], montgomery_sam1_all_avg[PredictTimeKey], bar_width, bottom=montgomery_sam1_all_avg[SetImageTimeKey], color=color_predict_time, zorder=10)

# Montgomery SAM2 all
bar11 = axes.bar(index[5], montgomery_sam2_all_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
bar12 = axes.bar(index[5], montgomery_sam2_all_avg[PredictTimeKey], bar_width, bottom=montgomery_sam2_all_avg[SetImageTimeKey], color=color_predict_time, zorder=10)

# # Montgomery SAM1 1by1
# bar13 = axes.bar(index[6], montgomery_sam1_1by1_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
# bar14 = axes.bar(index[6], montgomery_sam1_1by1_avg[PredictTimeKey], bar_width, bottom=montgomery_sam1_1by1_avg[SetImageTimeKey], color=color_predict_time, zorder=10)
#
# # Montgomery SAM2 1by1
# bar15 = axes.bar(index[7], montgomery_sam2_1by1_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
# bar16 = axes.bar(index[7], montgomery_sam2_1by1_avg[PredictTimeKey], bar_width, bottom=montgomery_sam2_1by1_avg[SetImageTimeKey], color=color_predict_time, zorder=10)
#
# # Montgomery SAM1 250
# bar17 = axes.bar(index[8], montgomery_sam1_250_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
# bar18 = axes.bar(index[8], montgomery_sam1_250_avg[PredictTimeKey], bar_width, bottom=montgomery_sam1_250_avg[SetImageTimeKey], color=color_predict_time, zorder=10)
#
# # Montgomery SAM2 250
# bar19 = axes.bar(index[9], montgomery_sam2_250_avg[SetImageTimeKey], bar_width, color=color_set_image_time, zorder=10)
# bar20 = axes.bar(index[9], montgomery_sam2_250_avg[PredictTimeKey], bar_width, bottom=montgomery_sam2_250_avg[SetImageTimeKey], color=color_predict_time, zorder=10)

# Primary X-axis without label
axes.set_xticks(index)
# axes.set_xticklabels(['SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2'])
axes.set_xticklabels(['SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2'])

# Secondary X-axis at the bottom without ticks and without top spine
ax2 = axes.secondary_xaxis(location=-0.075)
# ax2.set_xticks([0.5, 2.5, 4.5, 6.5, 8.5])
ax2.set_xticks([0.5, 2.5, 4.5])
# ax2.set_xticklabels(['\nCoronacases', '\nRadiopaedia', '\nMontgomery (all)', '\nMontgomery (1 by 1)', '\nMontgomery (250)'])
ax2.set_xticklabels(['Coronacases', 'Radiopaedia', 'Montgomery'])
ax2.xaxis.set_ticks_position('none')
ax2.spines['top'].set_visible(False)
# ax2.set_xlabel('Dataset')

# Y-axis
axes.set_ylabel('Average Time (s)')
if not paper_version:
    axes.set_title('Processing Time Comparison')
axes.legend()

plt.show()

# figure.savefig('sam2_delta_report_image/time_sam1_vs_sam2.pdf', bbox_inches='tight')
# figure.savefig('sam2_delta_report_image/time_sam1_vs_sam2.png', bbox_inches='tight')

time_sam1_vs_sam2_dictionary = {
    "coronacases_sam1_avg": coronacases_sam1_avg,
    "coronacases_sam2_avg": coronacases_sam2_avg,
    "radiopaedia_sam1_avg": radiopaedia_sam1_avg,
    "radiopaedia_sam2_avg": radiopaedia_sam2_avg,
    "montgomery_sam1_all_avg": montgomery_sam1_all_avg,
    "montgomery_sam2_all_avg": montgomery_sam2_all_avg,
    "montgomery_sam1_1by1_avg": montgomery_sam1_1by1_avg,
    "montgomery_sam2_1by1_avg": montgomery_sam2_1by1_avg,
    "montgomery_sam1_250_avg": montgomery_sam1_250_avg,
    "montgomery_sam2_250_avg": montgomery_sam2_250_avg
}
df = pd.DataFrame.from_dict(time_sam1_vs_sam2_dictionary, orient="index")
df.to_csv("sam2_delta_report_image/time_sam1_vs_sam2.csv")
