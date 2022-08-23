"""Submit a local notebook to kaggle"""

from copy import deepcopy
import os
import subprocess
import json
import shutil
import tempfile
import sys
import pprint
import click

NOTEBOOKS_DIR = 'kaggle_notebook' 
KERNEL_SOURCES = ['yanickjose/create-sample-dataset']
COMPETITION_SOURCES = ['g-research-crypto-forecasting']
KAGGLE_ACCOUNT = 'yanickjose'


META_DATA_TEMPLATE = {
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": False,
  "enable_internet": False,
  "competition_sources": COMPETITION_SOURCES,
}

def to_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def run_shell(command, verbose=True):
    print("Executing:", command)
    p = subprocess.Popen(
        [command], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    stdout, stderr = p.communicate()

    stdout = stdout.decode("utf-8")
    stderr = stderr.decode("utf-8")  # stderr contains warnings.

    if p.returncode != os.EX_OK:
        print("Return code:", p.returncode)
        print(stdout)
        print(stderr)
        raise sys.exit(p.returncode)

    if verbose:
        if stdout != "":
            print(stdout)

        if stderr != "":
            print(stderr)

    return p.returncode


def get_file_title(input_path: str) ->  str:
    title = os.path.basename(str(input_path)).split('.')[0] # get the title
    return title.lower()


def create_metadata(input_path: str, add_data_sources: bool = True,
                    add_kernel_sources: bool = True):
    metadata = deepcopy(META_DATA_TEMPLATE)
    title = get_file_title(input_path)
    metadata['id'] = f'{KAGGLE_ACCOUNT}/{title}'
    metadata['title'] = title
    if add_kernel_sources:
        metadata['kernel_sources'] = KERNEL_SOURCES
    return metadata


def push_to_kaggle(input_path: str, add_data_sources: bool = True,
                   add_kernel_sources: bool = True) -> None:
    metadata = create_metadata(input_path,
                               add_data_sources=add_data_sources,
                               add_kernel_sources=add_kernel_sources)
    file_name = os.path.basename(input_path)
    metadata['code_file'] = file_name

    with tempfile.TemporaryDirectory() as tmpdir:
        pprint.pprint(metadata)
        to_json(metadata, os.path.join(tmpdir, "kernel-metadata.json"))
         # Copy the target kernel to `tmpdir`.
        dst = os.path.join(tmpdir, file_name)
        shutil.copyfile(input_path, dst)

        # Push the kernel to Kaggle.
        run_shell(f"kaggle kernels push -p {tmpdir}")
        run_shell(f'kaggle kernels status {metadata["id"]}')


@click.command()
@click.option('-f', '--input_file', type=click.Path())
def cli(input_file):
    input_path = os.path.join(NOTEBOOKS_DIR, input_file)
    assert os.path.exists(input_path)
    push_to_kaggle(input_path)


if __name__ == '__main__':
    cli()


