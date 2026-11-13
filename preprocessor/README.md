# Preprocessor: Pipeline for IoT Traffic Datasets

This project provides a complete pipeline for preprocessing, analysis, feature selection, and reduction of IoT network traffic datasets. The goal is to standardize and efficiently prepare the data for use in Machine Learning and cybersecurity analysis tasks.

---

## Project Structure

The project is organized into a series of notebooks and configuration files that must be executed in sequential order.

-   **`config.json` / `config.yaml`**: Configuration files that define global parameters, dataset metadata, and runtime environment settings.
-   **`0_Source_Datasets_Analysis.ipynb`**: Performs an exploratory analysis of the source datasets. It generates descriptive statistics, category counts, and analysis of missing values.
-   **`1_Preprocessing_<Datasetname>.ipynb`**: Individual notebooks for the specific preprocessing of each dataset. These scripts transform the raw data into a standardized intermediate format (e.g., `1_Preprocessing_Bot-IoT-full.ipynb`).
-   **`2_Preprocessing_chunks_PARQUET.ipynb`**: Applies a general preprocessing to the standardized data. This step includes imputation of missing values, encoding of categorical variables, normalization, and splitting into training and testing sets. The results are saved in Parquet format for greater efficiency.
-   **`3_10percent_subset_stratified.ipynb`**: Creates 10% stratified subsets of each dataset. This is useful for faster experimentation and testing with a smaller volume of data.
-   **`4_Data_Structure.ipynb`**: This notebook has replaced the old `4_Feature_reduction.ipynb`. It is responsible for analyzing and possibly restructuring the data to optimize memory usage and efficiency in the following steps.
-   **`5_Feature_selection.ipynb`**: Implements feature selection methods to identify the most relevant ones. It uses statistical techniques such as `chi2`, `f_classif`, and `mutual_info_classif`.
-   **`README.md`**: This file.

---
## Configuration

The pipeline's configuration is managed through two main files: `config.json` and `config.yaml`.

### `config.json`
This file contains the metadata for each of the datasets to be processed. The configuration for each dataset is a JSON object where the key is the dataset name, and the value contains the following attributes:

-   `path`: The relative path to the dataset file.
-   `label`, `anomaly_label`, `normal_label`: The names of the label column and the values corresponding to anomalous and normal traffic.
-   `category_col`, `sub_category_col`: The columns containing the attack categories and subcategories.
-   `id_cols`: Identifier columns that should be excluded from the training process.
-   `numerical_cols`, `categorical_cols`: Lists of the numerical and categorical column names.
-   `timestamp_col`, `timestamp_format`: Information about the timestamp column and its format (e.g., "Unix_seconds").

The configured datasets include: `N-baiot`, `IoTID20`, `Bot-IoT`, `Bot-IoT-full`, `DS2OS`, `Ton-IoT`, `IOT23`, and `CICIoTDataset2023`.

### `config.yaml`
This file defines the execution environments, classifiers, and feature selection methods.

-   **Environments**: You can define multiple environments such as `high_performance`, `google_colab`, or `low_performance`, each with its own paths (`ROOT`, `RESULTS`) and parameters (`CHUNKSIZE`, `NTRIALS`).
-   **Classifiers**: Defines the Machine Learning models to be used, both supervised (`RandomForest`, `XGBoost`, etc.) and unsupervised (`IsolationForest`, `KMeans`, etc.).
-   **Feature Selection**: Specifies the methods to be used (`chi2`, `f_classif`, `mutual_info_classif`) and on which datasets to apply them.

---

## Workflow and Preprocessing Logic

The pipeline is designed to be executed sequentially, where the output of one step is the input of the next.

1.  **Initial Configuration**: Before running the pipeline, it is crucial to configure the `config.json` and `config.yaml` files. In `config.json`, define the metadata for each dataset you want to process. In `config.yaml`, select the execution environment (`environment_current`) and define the paths (`ROOT`, `RESULTS`) and other parameters like `CHUNKSIZE`.

2.  **Exploratory Data Analysis (EDA)**: Run `0_Source_Datasets_Analysis.ipynb` to get a deep understanding of the original data. This notebook generates reports that are essential for understanding the distribution and characteristics of each dataset.

3.  **Specific Preprocessing per Dataset**: For each dataset defined in the configuration, run its corresponding `1_Preprocessing_<Datasetname>.ipynb` notebook. This step is crucial for handling the particularities of each data source and standardizing them.

4.  **General Preprocessing**: Once all datasets have been standardized, run `2_Preprocessing_chunks_PARQUET.ipynb`. This script will apply a set of common transformations to all datasets, preparing them for modeling.

5.  **Creation of Subsets (Optional)**: If you need to perform quick tests, run `3_10percent_subset_stratified.ipynb` to generate smaller and more manageable versions of the datasets.

6.  **Data Structure Analysis**: Run `4_Data_Structure.ipynb` to analyze the structure of the preprocessed data.

7.  **Feature Selection**: Finally, run `5_Feature_selection.ipynb` to reduce the dimensionality of the datasets, selecting only the most informative features. This improves the efficiency and, often, the performance of Machine Learning models.

---

## Results

All artifacts generated by the pipeline, including processed datasets, subsets, analysis reports, and tables in LaTeX format, are saved in the directory specified in the `RESULTS` variable within the `config.yaml` file. The folder structure allows for easy access and organization of the results for their subsequent use in model training and evaluation.

---

## Requirements

-   Python 3.8+
-   Libraries: `pandas`, `numpy`, `scikit-learn`, `pyyaml`, `psutil`, `matplotlib`, `seaborn`, `pyarrow`.
-   Optional: `scapy` for processing PCAP files.

You can install the necessary dependencies by running:
```sh
pip install -r requirements.txt