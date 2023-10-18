#!usr/bin/python3

from typing import Sequence, Optional
import os

from argparse import ArgumentParser
from datasets import load_dataset
from huggingface_hub import login, HfFolder

"""
Author: Mark Simmons

Downloads a dataset at DATASET_URL from huggingface.
Check if user has token stored in cache, if not prompts user to login.
Saves dataset to LOCAL_PATH.
"""

def can_make_dir(parser: ArgumentParser, arg: str) -> str:
    """
    Return error if directory path not found, return filepath otherwise.
    """
    try:
        os.makedirs(arg, exist_ok=True)
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

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = ArgumentParser("Download HuggingFace Dataset")
    init_args(parser)
    args = parser.parse_args(argv)

    token = HfFolder.get_token()
    while not token:
        login()
        token = HfFolder.get_token()

    data = load_dataset(args.DATASET_URL)
    if args.type == 'CSV':
        data.to_csv(args.LOCAL_PATH)
    elif args.type == 'JSON':
        data.to_json(args.LOCAL_PATH)
    else:
        data.save_to_disk(args.LOCAL_PATH)

    return 0

if __name__ == '__main__':
    main()