import loguru
import requests
import pandas as pd

import mteb

logger = loguru.logger

SIZE__API_URL_BASE = "https://datasets-server.huggingface.co/size?dataset="
STATS__API_URL_BASE = "https://datasets-server.huggingface.co/statistics?dataset="


def query(url: str):
    try:
        logger.info(f"Querying {url}")
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        logger.error(err)
        return None
    return response.json()

def get_descriptive_stats_value(task, level0_key: str = "n_samples", split: str = "test", level1_key: str = None):
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
                        return "N/A"
                return n_samples_dict[split]
    return "N/A"

def get_dataset_stats(task: mteb.TaskMetadata):
    dataset_name = task.metadata.dataset["path"]
    data = query(url=STATS__API_URL_BASE+dataset_name+"&split=test&config=default")
    if data:
        if data["statistics"]:
            for stats in data["statistics"]:
                if stats["column_type"] == "class_label":
                    return stats["column_statistics"]["n_unique"]
    return "N/A"

def get_dataset_info(dataset: str):
    data = query(url=SIZE__API_URL_BASE+dataset)

    num_qrels = num_corpus_elements = num_queries = "N/A"
    if data:
        if data["size"]:
            if data['size']['splits']:
                for split in data['size']['splits']:
                    if split['config'] == 'qrels' and split['split'] == 'test':
                        num_qrels = split['num_rows']
                    elif split['config'] == 'corpus':
                        num_corpus_elements = split['num_rows']
                    elif split['config'] == 'query' and split['split'] == 'test':
                        num_queries = split['num_rows']
    return {
        'num_qrels': num_qrels,
        'num_corpus_elements': num_corpus_elements,
        'num_queries': num_queries
    }

def get_stats(task: mteb.TaskMetadata, key: str):
    task_name = task.metadata.name
    stats = dataset_info[task_name][key]
    map_keys = {
        "num_qrels": "num_qrels",
        "num_corpus_elements": "num_documents",
        "num_queries": "num_queries"
    }
    if stats == "N/A":
        return get_descriptive_stats_value(task, level0_key="avg_character_length", split="test", level1_key=map_keys[key])
    return stats

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

dataset_info = {
    task.metadata.name: get_dataset_info(task.metadata.dataset["path"])
    for task in tasks
}

# TODO: handle dev split
data = [
    {
        "Task": task.metadata.name,
        "Type": task.metadata.type,
        # "\# Samples Train": get_descriptive_stats_value(task, "n_samples", "train"),
        # "\# Samples Val": get_n_samples_per_split(task, "n_samples", "validation"),
        "\# Samples Test": get_descriptive_stats_value(task, "n_samples", "test"),
        "\# Queries": get_stats(task, "num_queries") if "Retrieval" in task.metadata.type else "N/A",
        # get_descriptive_stats_value(task, "avg_character_length", "test", "num_queries"),
        "\# Documents": get_stats(task, "num_corpus_elements") if "Retrieval" in task.metadata.type else "N/A",
        "\# Qrels": get_stats(task, "num_qrels") if "Retrieval" in task.metadata.type else "N/A", 
        "\# Labels": get_dataset_stats(task) if "Classification" in task.metadata.type else "N/A",
    }
    for task in tasks
]

df = pd.DataFrame.from_dict(data)
print(df.shape, df.columns, df.info())

df.set_index("Type", inplace=True)

column_format = ''.join(['c'] * (len(df.columns) + 1))

columns_per_task_type = {
    "Retrieval": ["Task", "\# Samples Test", "\# Queries", "\# Documents", "\# Qrels"],
    "Classification": ["Task", "\# Samples Test", "\# Labels"],
    "Clustering": ["Task", "\# Samples Test", "\# Labels"],
}

for type_ in df.index.unique():
    logger.info(f"Saving {type_} tasksk to LaTeX table.")
    selected_df = df.copy()
    
    if "Retrieval" in type_:
        selected_df = df[columns_per_task_type["Retrieval"]].copy()
    elif "Classification" in type_:
        selected_df = df[columns_per_task_type["Classification"]].copy()
    elif "Clustering" in type_:
        selected_df = df[columns_per_task_type["Clustering"]].copy()

    latex_code = selected_df.loc[type_].to_latex(
        f"mieb_datasets_{type_}.tex",
        # index=False,
        caption=f'Datasets overview and metadata for {type_} tasks.',
        label=f'tab:datasets_{type_}',
        column_format=column_format)
    