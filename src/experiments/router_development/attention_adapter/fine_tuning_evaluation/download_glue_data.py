"""Download official GLUE data.

SST-2 and RTE are active for the downstream evaluation suite right now. The
disabled mappings below are kept as TODOs so additional GLUE tasks can be
re-enabled one at a time later.
"""

from __future__ import annotations

import argparse
import io
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import zipfile


TASKS = [
    "SST",
    "RTE",
    # TODO: re-enable GLUE tasks as the evaluation suite grows.
    # "CoLA",
    # "MRPC",
    # "QQP",
    # "STS",
    # "MNLI",
    # "QNLI",
    # "WNLI",
    # "diagnostic",
]

TASK_ALIASES = {
    "sst2": "SST",
    "SST-2": "SST",
    "rte": "RTE",
}

TASK2PATH = {
    "SST": "https://dl.fbaipublicfiles.com/glue/data/SST-2.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    # TODO: future GLUE tasks, currently unsupported by fine_tuning_evaluation.
    # "CoLA": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    # "QQP": "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    # "STS": "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
    # "MNLI": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    # "QNLI": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
    # "WNLI": "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
    # "diagnostic": "https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
}

FUTURE_TASK2PATH = {
    "CoLA": "https://dl.fbaipublicfiles.com/glue/data/CoLA.zip",
    "QQP": "https://dl.fbaipublicfiles.com/glue/data/QQP-clean.zip",
    "STS": "https://dl.fbaipublicfiles.com/glue/data/STS-B.zip",
    "MNLI": "https://dl.fbaipublicfiles.com/glue/data/MNLI.zip",
    "QNLI": "https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip",
    "RTE": "https://dl.fbaipublicfiles.com/glue/data/RTE.zip",
    "WNLI": "https://dl.fbaipublicfiles.com/glue/data/WNLI.zip",
    "diagnostic": "https://dl.fbaipublicfiles.com/glue/data/AX.tsv",
}

MRPC_TRAIN = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_train.txt"
MRPC_TEST = "https://dl.fbaipublicfiles.com/senteval/senteval_data/msr_paraphrase_test.txt"


def canonical_task(task: str) -> str:
    return TASK_ALIASES.get(task, task)


def sst2_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "SST-2"


def rte_dir(data_dir: str | Path) -> Path:
    return Path(data_dir) / "RTE"


def sst2_files_exist(data_dir: str | Path) -> bool:
    task_dir = sst2_dir(data_dir)
    return all((task_dir / name).is_file() for name in ("train.tsv", "dev.tsv", "test.tsv"))


def rte_files_exist(data_dir: str | Path) -> bool:
    task_dir = rte_dir(data_dir)
    return all((task_dir / name).is_file() for name in ("train.tsv", "dev.tsv", "test.tsv"))


def download_and_extract(task: str, data_dir: str | Path) -> None:
    task = canonical_task(task)
    if task not in TASK2PATH:
        raise ValueError(f"Task {task!r} is not active; active tasks={TASKS}. Future tasks={sorted(FUTURE_TASK2PATH)}")
    print(f"Downloading and extracting {task}...")
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    data_file = Path(data_dir) / f"{task}.zip"
    urllib.request.urlretrieve(TASK2PATH[task], data_file)
    with zipfile.ZipFile(data_file) as zip_ref:
        zip_ref.extractall(data_dir)
    data_file.unlink()
    print("\tCompleted!")


def ensure_sst2(data_dir: str | Path) -> Path:
    if not sst2_files_exist(data_dir):
        download_and_extract("SST", data_dir)
    return sst2_dir(data_dir)


def ensure_rte(data_dir: str | Path) -> Path:
    if not rte_files_exist(data_dir):
        download_and_extract("RTE", data_dir)
    return rte_dir(data_dir)


def ensure_glue_task(data_dir: str | Path, task: str) -> Path:
    canonical = canonical_task(task)
    if canonical == "SST":
        return ensure_sst2(data_dir)
    if canonical == "RTE":
        return ensure_rte(data_dir)
    raise ValueError(f"Task {canonical!r} is not active; active tasks={TASKS}")


def format_mrpc(data_dir: str, path_to_data: str) -> None:
    """Preserved for future MRPC support; MRPC is not active right now."""
    print("Processing MRPC...")
    mrpc_dir = os.path.join(data_dir, "MRPC")
    if not os.path.isdir(mrpc_dir):
        os.mkdir(mrpc_dir)
    if path_to_data:
        mrpc_train_file = os.path.join(path_to_data, "msr_paraphrase_train.txt")
        mrpc_test_file = os.path.join(path_to_data, "msr_paraphrase_test.txt")
    else:
        try:
            mrpc_train_file = os.path.join(mrpc_dir, "msr_paraphrase_train.txt")
            mrpc_test_file = os.path.join(mrpc_dir, "msr_paraphrase_test.txt")
            urllib.request.urlretrieve(MRPC_TRAIN, mrpc_train_file)
            urllib.request.urlretrieve(MRPC_TEST, mrpc_test_file)
        except urllib.error.HTTPError:
            print("Error downloading MRPC")
            return
    assert os.path.isfile(mrpc_train_file), f"Train data not found at {mrpc_train_file}"
    assert os.path.isfile(mrpc_test_file), f"Test data not found at {mrpc_test_file}"

    with io.open(mrpc_test_file, encoding="utf-8") as data_fh, io.open(
        os.path.join(mrpc_dir, "test.tsv"), "w", encoding="utf-8"
    ) as test_fh:
        data_fh.readline()
        test_fh.write("index\t#1 ID\t#2 ID\t#1 String\t#2 String\n")
        for idx, row in enumerate(data_fh):
            _label, id1, id2, s1, s2 = row.strip().split("\t")
            test_fh.write(f"{idx}\t{id1}\t{id2}\t{s1}\t{s2}\n")

    print("\tMRPC conversion is preserved but disabled until MRPC is re-enabled.")


def get_tasks(task_names: str) -> list[str]:
    requested = [canonical_task(name.strip()) for name in task_names.split(",") if name.strip()]
    if "all" in requested:
        return TASKS
    tasks = []
    for task_name in requested:
        if task_name not in TASKS:
            raise ValueError(f"Task {task_name!r} is not active; active tasks={TASKS}")
        tasks.append(task_name)
    return tasks


def main(arguments: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="directory to save data to", type=str, default="glue_data")
    parser.add_argument("--tasks", help="comma-separated tasks to download", type=str, default="SST")
    parser.add_argument(
        "--path_to_mrpc",
        help="future MRPC support: directory containing msr_paraphrase_train.txt and msr_paraphrase_test.txt",
        type=str,
        default="",
    )
    args = parser.parse_args(arguments)

    Path(args.data_dir).mkdir(parents=True, exist_ok=True)
    for task in get_tasks(args.tasks):
        if task == "MRPC":
            format_mrpc(args.data_dir, args.path_to_mrpc)
        else:
            download_and_extract(task, args.data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
