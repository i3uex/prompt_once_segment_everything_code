"""
Analyzes the diversity of the COVID-19 and the Montgomery datasets, as one of
the reviewers asked us to analyze the diversity of the dataset.

The metadata file for the COVID-19 dataset is taken from the GitHub repository:
https://github.com/ieee8023/covid-chestxray-dataset

Our paper uses the dataset from: https://zenodo.org/records/3757476 It is a
subset of the COVID-19 dataset, including only its volumes:
https://academictorrents.com/details/136ffddd0959108becb2b3a86630bec049fcb0ff

The repository at Zenodo includes segmentation masks.

We will analyze only those entries in the folder "volumes", as those are the
ones that ended in the Zenodo repository.

The metadata for the Montgomery dataset is in the dataset itself, inside the
folder "NLM-MontgomeryCXRSet\MontgomerySet\ClinicalReadings". There is one text
file for each Xray image. Unfortunately, the download script
(scripts/download_montgomery_dataset.sh) deletes those files. For future
reference, they have been included in this folder.

We will read each file and return the data in a Pandas DataFrame, so it can be
plotted in the same way as the COVID-19 metadata.
"""

import json
import os
import re

import matplotlib.ticker as mtick
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tqdm import tqdm

from plots import common, constants

Covid19MetadataFilePath = "covid_19_metadata.csv"
MontgomeryMetadataFolderPath = "montgomery_metadata"
FolderKey = "folder"
SexKey = "sex"
AgeKey = "age"


# region Data loaders

def load_covid_19_data() -> pd.DataFrame:
    df = pd.read_csv(Covid19MetadataFilePath)
    volumes_df = df[df[FolderKey] == "volumes"]
    volumes_df = volumes_df[[SexKey, AgeKey]]
    volumes_df = volumes_df.astype({AgeKey: 'int'})

    return volumes_df


def extract_info_from_file(file_path):
    """Extracts patient"s sex and age from the first two lines of a given TXT file."""
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()
        if len(lines) < 2:
            return None, None  # Skip files with insufficient lines

        # Extract sex
        sex_match = re.search(r"Patient's Sex:\s*([MF])", lines[0])
        sex = sex_match.group(1) if sex_match else None

        # Extract age
        age_match = re.search(r"Patient's Age:\s*(\d{3})Y", lines[1])
        age = int(age_match.group(1)) if age_match else None

        return sex, age


def process_folder(folder_path):
    """Processes all TXT files in the given folder and collects sex and age data."""
    sex_data = []
    age_data = []

    file_list = os.listdir(folder_path)
    progress_bar = tqdm(desc="Getting mask sizes", total=len(file_list))
    for filename in file_list:
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            sex, age = extract_info_from_file(file_path)
            if sex is not None and age is not None:
                sex_data.append(sex)
                age_data.append(age)
        progress_bar.update()
    progress_bar.close()

    df = pd.DataFrame({
        SexKey: sex_data,
        AgeKey: age_data
    })

    return df


def load_montgomery_data() -> pd.DataFrame:
    return process_folder(MontgomeryMetadataFolderPath)


def save_data(dataset_name: str, df: pd.DataFrame):
    df.to_csv(f"../../paper/datasets_diversity/{dataset_name.lower()}.csv", index=False)

    sex_counts = df[SexKey].value_counts(normalize=True) * 100  # Get percentages
    sex_percentages = sex_counts.to_dict()
    with open(f"../../paper/datasets_diversity/{dataset_name.lower()}_sex_percentages.json", "w") as file:
        file.write(json.dumps(sex_percentages, indent=4))

# endregion

# region Plot creation

def create_sex_histogram(dataset_name: str, df: pd.DataFrame, paper_version: bool):
    common.setup_pyplot()
    figure_size = constants.PYPLOT_FIGURE_SIZE
    figure, axes = plt.subplots(figsize=figure_size, dpi=constants.PYPLOT_PLOT_DPI)

    total = len(df)

    # Normalize counts to percentages
    ax = sns.histplot(df, x=SexKey, stat="percent", discrete=True, shrink=0.8, edgecolor=None, alpha=1)
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.grid(linewidth=1, which='major')
    ax.grid(linewidth=0.5, which='minor')

    # for bar in ax.patches:
    #     count = int((bar.get_height() / 100) * total)  # Convert percentage back to count
    #     percentage = f"{bar.get_height():.2f} %"  # Display percentage
    #     ax.annotate(f"{count} ({percentage})",
    #                 xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
    #                 ha="center", va="bottom", fontsize=constants.PYPLOT_SIZE * 0.6)

    plt.ylim(0, 65)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Male", "Female"])
    plt.xlabel("Sex")
    plt.ylabel("Percentage of samples")

    if not paper_version:
        plt.title(f"Histogram of sex for {dataset_name}")

    plt.show()

    figure.savefig(f"../../paper/datasets_diversity/{dataset_name.lower()}_sex.pdf", bbox_inches="tight")
    figure.savefig(f"../../paper/datasets_diversity/{dataset_name.lower()}_sex.png", bbox_inches="tight")


def create_age_histogram(dataset_name: str, df: pd.DataFrame, paper_version: bool):
    common.setup_pyplot()
    figure_size = constants.PYPLOT_FIGURE_SIZE
    figure, axes = plt.subplots(figsize=figure_size, dpi=constants.PYPLOT_PLOT_DPI)

    total = len(df)

    # Normalize counts to percentages
    ax = sns.histplot(df[AgeKey], kde=False, edgecolor=None, alpha=1, stat="percent", bins=10)
    ax.set_axisbelow(True)
    ax.yaxis.set_minor_locator(mtick.AutoMinorLocator())
    ax.grid(linewidth=1, which='major')
    ax.grid(linewidth=0.5, which='minor')

    plt.xticks(range(10, 91, 10))
    plt.xlim(0, 92)
    plt.ylim(0, 27)

    # Add labels and title
    plt.xlabel("Age")
    plt.ylabel("Percentage of samples")  # Updated label
    if not paper_version:
        plt.title(f"Histogram of age for {dataset_name}")

    plt.show()

    figure.savefig(f"../../paper/datasets_diversity/{dataset_name.lower()}_age.pdf", bbox_inches="tight")
    figure.savefig(f"../../paper/datasets_diversity/{dataset_name.lower()}_age.png", bbox_inches="tight")


# endregion

def main():
    paper_version = True

    df = load_covid_19_data()
    create_sex_histogram("COVID-19", df, paper_version)
    create_age_histogram("COVID-19", df, paper_version)
    save_data("COVID-19", df)

    df = load_montgomery_data()
    create_sex_histogram("Montgomery", df, paper_version)
    create_age_histogram("Montgomery", df, paper_version)
    save_data("Montgomery", df)


if __name__ == "__main__":
    main()
