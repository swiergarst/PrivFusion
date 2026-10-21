## Main entrypoint, combining federated harmonization and local modifications
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from tqdm import tqdm

from local_modifications import CSVAgent
from privfusion.agents.llms import WillmaLLM
from privfusion.consolidater import Consolidator
from privfusion.dataset_analyzer import DatasetAnalyzer
from setup_logging import consoleLogger

FMT = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")


def federated_harmonization(datasets_in: list, run_id: str, round: int):
    # llm = WillmaLLM(model_name=os.environ.get("HARM_LLM_MODEL"))
    # analyzer = DatasetAnalyzer(llm)
    # datasets = {}

    # for dataset_in in datasets_in:
    #     dataset_name = dataset_in["name"]
    #     dataset = pd.read_csv(dataset_in["path"])

    #     consoleLogger.info(f"extracting information from dataset {dataset_name}..")
    #     dataset_info = analyzer.extract_information_from_dataset(dataset_name, dataset)
    #     datasets[dataset_name] = {"data": dataset, "info": dataset_info}

    # consolidator = Consolidator()
    # consoleLogger.info("consolidating..")
    # result = consolidator.consolidate(datasets, llm)
    # return result

    dataset_names = []
    dataset_paths = []
    for dataset in datasets_in:
        dataset_names.append(dataset["name"])
        dataset_paths.append(dataset["path"])

    return central_fn(dataset_names, dataset_paths)


# in actual application, both of these should be env variables (or data path is explicitly set in e.g. v6)
def local_client_fn(dataset_name: str, data_path: str) -> dict:
    llm = WillmaLLM(model_name=os.environ.get("HARM_LLM_MODEL"))
    analyzer = DatasetAnalyzer(llm)
    dataset = pd.read_csv(data_path)
    consoleLogger.info(f"extracting information from dataset {dataset_name}..")
    dataset_info = analyzer.extract_information_from_dataset(dataset_name, dataset)

    return {"dataset_name": dataset_name, "info": dataset_info}


# see local fn: these should both disappear in actual application
def central_fn(dataset_names: list, dataset_paths: list):
    round_results = {}
    for dataset_name, dataset_path in zip(dataset_names, dataset_paths, strict=False):
        client_result = local_client_fn(dataset_name, dataset_path)
        round_results[client_result["dataset_name"]] = client_result

    llm = WillmaLLM(model_name=os.environ.get("HARM_LLM_MODEL"))
    consolidator = Consolidator()
    consoleLogger.info("consolidating..")
    consoliator_result = consolidator.consolidate(round_results, llm)

    return consoliator_result


def set_round_handler(logger_name: str, filename: str) -> None:
    """Replace any handler previously installed by this helper with a new FileHandler."""
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # avoid duplicates via root

    # remove + close handlers from the previous round
    for h in list(logger.handlers):
        if getattr(h, "_per_round", False):
            logger.removeHandler(h)
            h.close()

    handler = logging.FileHandler(filename)
    handler.setFormatter(FMT)
    handler._per_round = True  # marker so we only remove our own handlers
    logger.addHandler(handler)


def setup_round_logging(run_id: str, round: int) -> None:
    set_round_handler("localModifLogger", f"logs/local_modif_logs_round{round}_id_{run_id}.log")
    set_round_handler("fedHarmonizationLogger", f"logs/fed_harmonization_logs_round{round}_id_{run_id}.log")


def main(config_path: str = "config.yaml"):
    run_id = time.time()
    with open(config_path) as fp:
        config = yaml.safe_load(fp)

    datasets_config = config["dataset_config"]

    for round in tqdm(range(config["rounds"])):
        setup_round_logging(run_id, round)
        federated_result = federated_harmonization(datasets_config, run_id, round)

        if config["save_intermediate"]:
            federated_result.to_csv(
                f"dataset_modifications/harmonization_instructions_round_{round}_{run_id}.csv", index=False
            )

        consoleLogger.info("finished consolidating, starting local modifications..")
        # TODO: this should be separated across clients
        new_dataset_config = []
        for dataset_config in datasets_config:
            dataset_name = dataset_config["name"]
            dataset_path = dataset_config["path"]
            instructions = federated_result.loc[federated_result["dataset"] == dataset_name]
            agent = CSVAgent()
            local_result = agent.run(dataset_name, dataset_path, instructions)
            if config["save_intermediate"]:
                with open(f"logs/local_results_{dataset_name}_round_{round}_id_{run_id}.json", "w") as fp:
                    # output_csv is the modified DataFrame (written as checkpoint); not JSON-serializable
                    json.dump({k: v for k, v in asdict(local_result).items() if k != "output_csv"}, fp, indent=2)

            if not local_result.success:
                consoleLogger.error(f"Agent failed: {local_result.error}")
                consoleLogger.error(f"({len(local_result.steps)} step(s) had been committed; nothing was written.)")
                raise SystemExit(1)

            consoleLogger.info(
                f"Replay check: {'PASS' if local_result.replay_ok else 'FAIL'} - {local_result.replay_message}"
            )

            new_dataset_path = (
                Path(config["dataset_checkpoints_path"]) / f"{dataset_name}_modified_round{round}_{run_id}.csv"
            )
            local_result.output_csv.to_csv(new_dataset_path)
            local_dataset_config = {"name": dataset_name, "path": new_dataset_path}
            new_dataset_config.append(local_dataset_config)

            if config["save_intermediate"]:
                script_path = Path(config["script_path"]) / f"{dataset_name}_reproduce_round_{round}_{run_id}.py"
                script_path.write_text(local_result.script)

        # TODO: implement early stopping if there are no modifications in the new round

        datasets_config = new_dataset_config


if __name__ == "__main__":
    load_dotenv()
    config = "config.yaml"
    main(config)
