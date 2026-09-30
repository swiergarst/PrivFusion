import pandas as pd

from privfusion.agents.llms import WillmaLLM
from privfusion.consolidater import Consolidator
from privfusion.data_models import DatasetInformation, SemanticInformation, StructuralInformation
from privfusion.dataset_analyzer import DatasetAnalyzer

dataset1 = pd.read_csv("data/covid_global.csv")
dataset2 = pd.read_csv("data/covid_indonesia.csv")


print(dataset1.head())
print(dataset2.head())

llm = WillmaLLM(model_name="openai/gpt-oss-120b")
analyzer = DatasetAnalyzer(llm)


print("analysing datasets..")

# dataset1_info = analyzer.extract_information_from_dataset("covid", dataset1)
# dataset2_info = analyzer.extract_information_from_dataset("covid", dataset2)

# datasets = {
#     "dataset1": {"data": dataset1, "info": dataset1_info},
#     "dataset2": {"data": dataset2, "info": dataset2_info},
# }


mock_semantics = SemanticInformation(description="", column_types=[], column_semantic=[], dataset_topics=[])
mock_structural = StructuralInformation(names=[], types=[], constraints=[], description="")
mock_info = DatasetInformation(name="mock", semantic=mock_semantics, structure=mock_structural, relationships=[])

datasets = {
    "dataset1": {"data": dataset1, "info": mock_info},
    "dataset2": {"data": dataset2, "info": mock_info},
}


print("datasets analysed, running consolidator..")

# Run consolidation
consolidator = Consolidator()
result = consolidator.consolidate(datasets, llm)

# View results
print(result[["dataset", "feature_name", "cluster_id", "norm_feature_name"]])

# print(dataset1_info)


# ###
# import pandas as pd
# import yaml

# from privfusion.agents.llms import WillmaLLM
# from privfusion.consolidater import Consolidator
# from privfusion.dataset_analyzer import DatasetAnalyzer
# from privfusion.data_models import DatasetInformation, SemanticInformation, StructuralInformation

# # with open('configs/experiment_1.yaml', "r") as fp:
# config = yaml.load(open('configs/experiment_1.yaml', "r"), Loader=yaml.Loader)


# dataset1 = pd.read_csv("data/covid_global.csv")
# dataset2 = pd.read_csv("data/covid_indonesia.csv")


# print(dataset1.head())
# print(dataset2.head())

# llm = WillmaLLM(model_name="openai/gpt-oss-120b")
# analyzer = DatasetAnalyzer(llm)


# dataset1_info = analyzer.extract_information_from_dataset("covid-global", dataset1)
# dataset2_info = analyzer.extract_information_from_dataset("covid-indonesia", dataset2)

# # with open("dataset1_info.txt", "w") as fp:
# #     fp.write(dataset1_info)


# # with open("dataset2_info.txt", "w") as fp:
# #     fp.write(dataset2_info)

# mock_semantics = SemanticInformation(description = "", column_types= [], column_semantic= [],dataset_topics=[])
# mock_structural = StructuralInformation(names = [], types= [], constraints = [], description = "")
# mock_info = DatasetInformation(name = "mock", semantic=mock_semantics, structure=mock_structural, relationships= [])


# datasets = {
#     "dataset1": {"data": dataset1, "info": dataset1_info},
#     "dataset2": {"data": dataset2, "info": dataset2_info},
# }

# # datasets = {
# #     "dataset1": {"data": dataset1, "info": mock_info},
# #     "dataset2": {"data": dataset2, "info": mock_info},
# # }

# # Run consolidation
# consolidator = Consolidator()
# result = consolidator.consolidate(datasets, llm)

# # View results
# print(result[["dataset", "feature_name", "cluster_id", "norm_feature_name"]])

# # print(dataset1_info)
