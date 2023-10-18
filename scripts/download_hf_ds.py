#!usr/bin/python3

from typing import Sequence, Optional, Literal, Union
import os

from argparse import ArgumentParser
from datasets import load_dataset, DatasetDict, Dataset
from huggingface_hub import login, HfFolder

"""
Author: Mark Simmons

Downloads a dataset at DATASET_URL from huggingface.
Check if user has token stored in cache, if not prompts user to login.
Saves dataset to LOCAL_PATH.
"""

def can_make_dir(parser: ArgumentParser, arg: str) -> str:
    """
    Return error if directory path cannot be made, return filepath otherwise.
    """
    try:
        os.makedirs(arg, exist_ok=True)
        return arg
    except Exception as e:
        parser.error(f"Could not make folder {arg}.", e)


def init_args(parser: ArgumentParser) -> None:
    add_arg = parser.add_argument
    add_arg(
        'DATASET_URL',
        help='Huggingface url for dataset to be downloaded (of shape USERNAME/DATASETNAME).'
    )
    add_arg(
        'LOCAL_PATH',
        help='Path for dataset to be stored locally.',
        type=lambda x: can_make_dir(parser, x)
    )
    add_arg(
        '--type',
        '-t',
        choices=['CSV', 'JSON'],
        help='Format to save dataset to. If None, use dataset.save_to_disk().'
    )

def save_dataset(
        data: Union[DatasetDict, Dataset],
        path: str,
        file_type: Literal['JSON', 'CSV', None] = None,
    ) -> str:
    if file_type is None:
        data.save_to_disk(path)
        return path
    if type(data) is Dataset:
        if file_type=='JSON':
            data.to_json(path)
        elif file_type=='CSV':
            data.to_csv(path)
        else:
            raise ValueError(f"file_type must be 'JSON', 'CSV' or None, {file_type=}")
        return path
    else:
        assert type(data) is DatasetDict
        return save_datasetdict(data, path, file_type)

def save_datasetdict(
        data: DatasetDict,
        path: str,
        file_type: Literal['JSON', 'CSV'],
    ) -> str:
    for split, split_data in data.items():
        split_path = os.path.join(path, split)
        print(f"Saving {split=} to {split_path}")
        if file_type=='JSON':
            split_data.to_json(split_path)
        elif file_type=='CSV':
            split_data.to_csv(split_path)
        else:
            raise ValueError(f"file_type must be 'JSON' or 'CSV', {file_type=}")


    return path

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = ArgumentParser("Download HuggingFace Dataset")
    init_args(parser)
    args = parser.parse_args(argv)

    token = HfFolder.get_token()
    while not token:
        login()
        token = HfFolder.get_token()

    data = load_dataset(args.DATASET_URL)
    save_dataset(data, args.LOCAL_PATH, args.type)

    return 0

if __name__ == '__main__':
    main()