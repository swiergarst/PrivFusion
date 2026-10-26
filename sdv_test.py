import warnings

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer

dataset_name = "covid-global"
dataset = pd.read_csv("datasets/covid_indonesia.csv")

# print(dataset.isna().sum())
# print(dataset.shape)
# print(dataset.dropna(how="all").shape)


synthetic_df = pd.DataFrame()
num_rows = 5
for column in dataset.columns:
    # print(dataset[[column]].shape)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        col_df = dataset[[column]]
        if col_df.isna().sum().sum() == col_df.shape[0]:  # column has only NaN's
            synthetic_df[column] = col_df.loc[:num_rows]
            continue
        metadata = Metadata.detect_from_dataframes(data={column: dataset[[column]].dropna()})
        synth = GaussianCopulaSynthesizer(metadata)
        synth.fit({column: dataset[[column]].dropna()})
        synthetic_column = synth.sample(num_rows=num_rows)[column]
        synthetic_df[column] = synthetic_column

# synthetic_df = pd.DataFrame(synthetic_dict)

print(synthetic_df.head())
