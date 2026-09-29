import pandas as pd

from privfusion.agents.llms import WillmaLLM
from privfusion.consolidater import Consolidator
from privfusion.dataset_analyzer import DatasetAnalyzer

dataset1 = pd.read_csv("data/covid_global.csv")
dataset2 = pd.read_csv("data/covid_indonesia.csv")


print(dataset1.head())
print(dataset2.head())

llm = WillmaLLM(model_name="openai/gpt-oss-120b")
analyzer = DatasetAnalyzer(llm)


dataset1_info = analyzer.extract_information_from_dataset("covid", dataset1)
dataset2_info = analyzer.extract_information_from_dataset("covid", dataset2)

datasets = {
    "dataset1": {"data": dataset1, "info": dataset1_info},
    "dataset2": {"data": dataset2, "info": dataset2_info},
}


# Run consolidation
consolidator = Consolidator()
result = consolidator.consolidate(datasets, llm)

# View results
print(result[["dataset", "feature_name", "cluster_id", "norm_feature_name"]])

# print(dataset1_info)
