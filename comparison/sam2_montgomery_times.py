import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from plots import common, constants

paper_version = True

# Load the datasets
file_paths = {
    'montgomery_sam1_all': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam1_all/joint_raw_data_2024-08-08_10-25-06.csv',
    'montgomery_sam1_1by1': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam1_1by1/joint_raw_data_2024-08-07_16-51-27.csv',
    'montgomery_sam1_250': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam1_250/joint_raw_data_2024-08-07_21-24-34.csv',
    'montgomery_sam2_all': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam2_all/joint_raw_data_2024-08-08_10-36-28.csv',
    'montgomery_sam2_1by1': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam2_1by1/joint_raw_data_2024-08-07_20-39-18.csv',
    'montgomery_sam2_250': '/mnt/c/Users/andy/OneDrive - Universidade de Santiago de Compostela/Documents/i3lab/sam2/working_data/montgomery/results/sam2_250/joint_raw_data_2024-08-07_21-40-30.csv',
}

datasets = {name: pd.read_csv(path) for name, path in file_paths.items()}

# Prepare data for each dataset
montgomery_sam1_all = datasets['montgomery_sam1_all'][['set_image_time', 'predict_time']].dropna()
montgomery_sam2_all = datasets['montgomery_sam2_all'][['set_image_time', 'predict_time']].dropna()
montgomery_sam1_1by1 = datasets['montgomery_sam1_1by1'][['set_image_time', 'predict_time']].dropna()
montgomery_sam2_1by1 = datasets['montgomery_sam2_1by1'][['set_image_time', 'predict_time']].dropna()
montgomery_sam1_250 = datasets['montgomery_sam1_250'][['set_image_time', 'predict_time']].dropna()
montgomery_sam2_250 = datasets['montgomery_sam2_250'][['set_image_time', 'predict_time']].dropna()

# Calculate the average set_image_time and predict_time for all datasets
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
# index = [0, 1, 2, 3, 4, 5]  # Bars for Coronacases SAM1, SAM2, Radiopaedia SAM1, SAM2, Montgomery SAM1, SAM2
index = [0, 1, 2, 3]  # Bars for Coronacases SAM1, SAM2, Radiopaedia SAM1, SAM2, Montgomery SAM1, SAM2

# Colors for consistency
color_set_image_time = 'orange'
color_predict_time = 'red'

# Montgomery SAM1 all
bar9 = axes.bar(index[0], montgomery_sam1_all_avg['set_image_time'], bar_width, color=color_set_image_time, zorder=10)
bar10 = axes.bar(index[0], montgomery_sam1_all_avg['predict_time'], bar_width, bottom=montgomery_sam1_all_avg['set_image_time'], color=color_predict_time, zorder=10)

# Montgomery SAM2 all
bar11 = axes.bar(index[1], montgomery_sam2_all_avg['set_image_time'], bar_width, color=color_set_image_time, label='Set Image Time', zorder=10)
bar12 = axes.bar(index[1], montgomery_sam2_all_avg['predict_time'], bar_width, bottom=montgomery_sam2_all_avg['set_image_time'], color=color_predict_time, label='Predict Time', zorder=10)

# # Montgomery SAM1 1by1
# bar13 = axes.bar(index[2], montgomery_sam1_1by1_avg['set_image_time'], bar_width, color=color_set_image_time, zorder=10)
# bar14 = axes.bar(index[2], montgomery_sam1_1by1_avg['predict_time'], bar_width, bottom=montgomery_sam1_1by1_avg['set_image_time'], color=color_predict_time, zorder=10)
#
# # Montgomery SAM2 1by1
# bar15 = axes.bar(index[3], montgomery_sam2_1by1_avg['set_image_time'], bar_width, color=color_set_image_time, zorder=10)
# bar16 = axes.bar(index[3], montgomery_sam2_1by1_avg['predict_time'], bar_width, bottom=montgomery_sam2_1by1_avg['set_image_time'], color=color_predict_time, zorder=10)

# Montgomery SAM1 250
bar17 = axes.bar(index[2], montgomery_sam1_250_avg['set_image_time'], bar_width, color=color_set_image_time, zorder=10)
bar18 = axes.bar(index[2], montgomery_sam1_250_avg['predict_time'], bar_width, bottom=montgomery_sam1_250_avg['set_image_time'], color=color_predict_time, zorder=10)

# Montgomery SAM2 250
bar19 = axes.bar(index[3], montgomery_sam2_250_avg['set_image_time'], bar_width, color=color_set_image_time, zorder=10)
bar20 = axes.bar(index[3], montgomery_sam2_250_avg['predict_time'], bar_width, bottom=montgomery_sam2_250_avg['set_image_time'], color=color_predict_time, zorder=10)

# Primary X-axis without label
axes.set_xticks(index)
# axes.set_xticklabels(['SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2'])
# axes.set_xticklabels(['SAM', 'SAM 2', 'SAM', 'SAM 2', 'SAM', 'SAM 2'])
axes.set_xticklabels(['SAM', 'SAM 2', 'SAM', 'SAM 2'])

# Secondary X-axis at the bottom without ticks and without top spine
ax2 = axes.secondary_xaxis(location=-0.075)
# ax2.set_xticks([0.5, 2.5, 4.5, 6.5, 8.5])
ax2.set_xticks([0.5, 2.5])
# ax2.set_xticklabels(['\nCoronacases', '\nRadiopaedia', '\nMontgomery (all)', '\nMontgomery (1 by 1)', '\nMontgomery (250)'])
# ax2.set_xticklabels(['\nOnly once', '\nEach time', '\nSame image'])
ax2.set_xticklabels(['Only once', 'Every time'])
ax2.xaxis.set_ticks_position('none')
ax2.spines['top'].set_visible(False)
# ax2.set_xlabel('Dataset')

# Y-axis
if not paper_version:
    axes.set_ylabel('Average Time (s)')
    axes.set_title('Montgomery Processing Time Comparison by Model Usage')
else:
    plt.tick_params(labelleft=False)
axes.legend()

plt.show()

figure.savefig('sam2_delta_report_image/time_operation_mode.pdf', bbox_inches='tight')
figure.savefig('sam2_delta_report_image/time_operation_mode.png', bbox_inches='tight')
