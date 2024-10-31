from typing import Optional

import loguru
import requests
import pandas as pd

from datasets import load_dataset

import mteb

logger = loguru.logger

SIZE__API_URL_BASE = "https://datasets-server.huggingface.co/size?dataset="
STATS__API_URL_BASE = "https://datasets-server.huggingface.co/statistics?dataset="


def query(url: str) -> Optional[str]:
    try:
        logger.info(f"Querying {url}")
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(err)
        return None
    return response.json()

def get_descriptive_stats_from_task_metadata(task, level0_key: str = "n_samples", split: str = "test", level1_key: str = None):
    split = split.lower()
    if task.metadata.descriptive_stats:
        n_samples_dict = task.metadata.descriptive_stats[level0_key]
        if n_samples_dict:
            split_keys = [key.lower() for key in n_samples_dict.keys()]
            use_validation = (
                split == "test" and 
                any(s in ["val", "validation", "dev"] for s in split_keys)
            )
            if split in split_keys or use_validation:
                if use_validation :
                    split = next(iter(set(split_keys) & {"val", "validation", "dev"})) # get the key that exists
                if isinstance(n_samples_dict[split], dict):
                    if level1_key and level1_key in n_samples_dict[split]:
                        return n_samples_dict[split][level1_key]
                    else:
                        return None
                return n_samples_dict[split]
    return None

def get_avg_choices_from_hf(task: mteb.TaskMetadata, split: str = "test"):
    logger.info(f"Getting average choices for {task.metadata.name}")
    data = load_dataset(task.metadata.dataset["path"])
    if "choices" in data[split].features:
        lengths = [len(choices) for choices in data[split]["choices"]]
        return sum(lengths) / len(lengths)
    return None

def get_dataset_stats_from_dataset_viewer(task: mteb.TaskMetadata, split: str = "test", config: str = "default"):
    dataset_name = task.metadata.dataset["path"]
    data = query(url=STATS__API_URL_BASE+dataset_name+f"&split={split}&config={config}")
    if data:
        if data["statistics"]:
            for stats in data["statistics"]:
                if stats["column_type"] == "class_label":
                    return stats["column_statistics"]["n_unique"]
    return None

def get_n_rows_from_dataset_viewer(task: mteb.TaskMetadata, split: str = None, config: str = "default"):
    data = query(url=SIZE__API_URL_BASE+task.metadata.dataset["path"])
    if data:
        if data["size"]:
            if data['size']['splits']:
                for split_ in data['size']['splits']:
                    if split:
                        if split_["config"] == config and split_['split'] == split:
                            return int(split_['num_rows'])
                    else:
                        if split_["config"] == config:
                            return int(split_['num_rows'])
    return None

# def get_retrieval_dataset_info_from_dataset_viewer(dataset: str):
#     data = query(url=SIZE__API_URL_BASE+dataset)

#     num_qrels = num_corpus_elements = num_queries = None
#     if data:
#         if data["size"]:
#             if data['size']['splits']:
#                 for split in data['size']['splits']:
#                     if split['config'] == 'qrels' and split['split'] == 'test':
#                         num_qrels = split['num_rows']
#                     elif split['config'] == 'corpus':
#                         num_corpus_elements = split['num_rows']
#                     elif split['config'] == 'query' and split['split'] == 'test':
#                         num_queries = split['num_rows']
#     return {
#         'num_qrels': num_qrels,
#         'num_corpus_elements': num_corpus_elements,
#         'num_queries': num_queries
#     }

# def get_stats(task: mteb.TaskMetadata, key: str):
#     global dataset_info

#     task_name = task.metadata.name
#     stats = dataset_info[task_name][key]
#     map_keys = {
#         "num_qrels": "num_qrels",
#         "num_corpus_elements": "num_documents",
#         "num_queries": "num_queries"
#     }
#     if stats is None:
#         return get_descriptive_stats_from_task_metadata(task, level0_key="avg_character_length", split="test", level1_key=map_keys[key])
#     return stats

tasks = mteb.get_tasks(
        task_types=[
            "Any2AnyRetrieval",
            "AbsTaskAny2AnyMultiChoice",
            "Any2TextMutipleChoice",
            "ImageClustering",
            "ImageClassification",
            "ImageMultilabelClassification",
            "ImageTextPairClassification",
            "VisualSTS",
            "ZeroShotClassification",
        ]
    )

all_metadata = [task.metadata for task in tasks]

task_names = [meta.dataset["path"] for meta in all_metadata]

# dataset_info = {
#     task.metadata.name: get_retrieval_dataset_info_from_dataset_viewer(task.metadata.dataset["path"])
#     for task in tasks
# }

def compute_value_from_hf(task: mteb.TaskMetadata, value: str = "n_samples", split: str = "test", config: str = "default"):
    logger.info(f"Getting {value} from HF")
    try:
        logger.info(f"Loading dataset {task.metadata.dataset['path']}..")
        data = load_dataset(
            task.metadata.dataset["path"], 
            split=split, name=config,
            trust_remote_code=True)
        match value:
            case "n_samples":
                return len(data)
            case "n_labels":
                label_column = set(data.column_names) & {"label", "labels", "choice", "choices", "class", "classes"}
                label = next(iter(label_column), None)
                if label:
                    return len(data.unique(label))
                logger.warning(f"Could not find label column in {task.metadata.name}")
                return None
            case _:
                logger.error(f"Value {value} not supported.")
                return None
    except ValueError as e:
        logger.error(e)
        return None


def reconcile_values(value_from_dataset_viewer, value_from_metadata):
    """Reconcile values from dataset viewer and metadata.
    If the value from the dataset viewer is None, return the value from the metadata.
    If the values are different, return the value from the dataset viewer.
    """
    if value_from_dataset_viewer is None:
        return value_from_metadata
    if value_from_dataset_viewer != value_from_metadata:
        return value_from_dataset_viewer
    return value_from_metadata


# TODO: handle dev split
data = [
    {
        "Task": task.metadata.name,
        "Type": task.metadata.type,
        "\# Samples Train": reconcile_values(
            get_n_rows_from_dataset_viewer(task, "train", "default"),
            get_descriptive_stats_from_task_metadata(task, "n_samples", "train")
        ) if "Classification" in task.metadata.type else None,
        "\# Samples Test": reconcile_values(
            get_n_rows_from_dataset_viewer(task, "test", "default"),
            get_descriptive_stats_from_task_metadata(task, "n_samples", "test")
        ) if "Any2Any" not in task.metadata.type else None,
        "\# Queries": reconcile_values(
            get_n_rows_from_dataset_viewer(task, split="test", config="query"), 
            get_descriptive_stats_from_task_metadata(task, "n_samples", "test", "num_queries")
        ) if "Any2Any" in task.metadata.type else None,
        "\# Documents": reconcile_values(
            get_n_rows_from_dataset_viewer(task, config="corpus"),
            get_descriptive_stats_from_task_metadata(task, "n_samples", "test", "num_corpus_elements")
        ) if "Any2Any" in task.metadata.type else None,
        "\# Qrels": reconcile_values(
            get_n_rows_from_dataset_viewer(task, split="test", config="qrels"),
            get_descriptive_stats_from_task_metadata(task, "n_samples", "test", "num_qrels")
        ) if "Any2Any" in task.metadata.type else None,
        "\# Labels": get_dataset_stats_from_dataset_viewer(task) if "Classification" in task.metadata.type else None,
        "Avg. \# Choices": get_avg_choices_from_hf(task) if "Choice" in task.metadata.type else None,
        "Metric": task.metadata.main_score,
    }
    for task in tasks
]

df = pd.DataFrame.from_dict(data)
df.fillna("-", inplace=True)
print(df.shape, df.columns, df.info())

df.set_index("Type", inplace=True)

column_format = ''.join(['c'] * (len(df.columns) + 1))

metric_values = {
    "accuracy": "Accuracy",
    "ndcg_at_10": "NDCG@10",
    "ndcg_at_5": "NDCG@5",
    "cv_recall_at_1": "Recall@1",
    "cosine_spearman": "Cosine Spearman",
    "text_acc": "Text Accuracy",
    "nmi": "NMI",
}

columns_per_task_type = {
    "Any2Any": ["Task", "\# Queries", "\# Documents", "\# Qrels", "Avg. \# Choices", "Metric"],
    "Other": ["Task", "\# Samples Train", "\# Samples Test", "\# Labels", "Metric"],
}

for type_ in df.index.unique():
    logger.info(f"Saving {type_} tasks to LaTeX table.")
    selected_df = df.copy()
    
    if "Any2Any" in type_:
        selected_df = df[columns_per_task_type["Any2Any"]].copy()
    else:
        selected_df = df[columns_per_task_type["Other"]].copy()

    selected_df['Metric'] = selected_df['Metric'].map(metric_values)

    latex_code = selected_df.loc[type_].to_latex(
        f"mieb_datasets_{type_}.tex",
        index=False,
        caption=f'Datasets overview and metadata for {type_} tasks.',
        label=f'tab:datasets_{type_}',
        column_format=column_format)
    