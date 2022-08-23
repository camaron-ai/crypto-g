"""
Import python modules to a notebook
"""
from pathlib import Path
from typing import List, Optional
from nbformat.v4 import new_code_cell
import nbformat
import os
import click
import git
import datetime
import pprint


current_dir = os.path.dirname(__file__)
INPUT_NOTEBOOKS_DIR = 'notebook' 
OUTPUT_NOTEBOOKS_DIR = 'kaggle_notebook' 


def get_info_message(message: str) -> None:
    repo = git.Repo(os.getcwd())
    info = {}

    master = repo.head.reference
    info['branch_name'] = master.name
    info['commit'] = master.commit.hexsha
    info['commit_msg'] = master.commit.message
    info['message'] = message
    info['lastest_commit_date'] = datetime.datetime.fromtimestamp(master.commit.committed_date).strftime('%Y-%m-%d %H:%M:%S')
    info['date'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return '\n'.join([f'# {key.upper()}: {item}' for key, item in info.items()])

    
def read_code(file: str) -> str:
    raw_code = Path(file).read_text()
    lines = raw_code.split('\n')
    lines = [line for line in lines if not line.endswith('#NO_IMPORT')]

    lines = [f'# IMPORTED FROM {file}'] + lines
    code = '\n'.join(lines)
    return code


def replace_code(input_path: str, modules: List[str], message) -> nbformat.notebooknode.NotebookNode:
    """Create a Jupyter notebook from text files and Python scripts."""
    nb = nbformat.read(input_path, as_version=4)

    modules = [os.path.join(current_dir, name)
                 for name in modules]

    lib_code = '\n\n\n'.join([read_code(file) for file in modules])

    info_msg = get_info_message(message)
    head_cell = new_code_cell(info_msg)
    cell_found = False
    for cell in nb['cells']:    # look for the cell to replace
        cell_code = cell['source']
        if cell_code.startswith('#IMPORT_SCRIPT!'):
            cell['source'] = lib_code
            cell_found = True
            break
    assert cell_found, 'cell starting with #IMPORT_SCRIPT! not found'
    nb['cells'].insert(0, head_cell)
    return nb


def submodule_files(submodule: str,
                    files: Optional[List[str]] = None) -> List[str]:
    if files is None:
       submodule_files = os.listdir(os.path.join(current_dir, submodule))
       files = sorted([name for name in submodule_files
                      if name.endswith(".py")])

    return [os.path.join(submodule, name)
            for name in files]


# pipeline files in order
pipeline_modules = ['feature_gen.py',
                  'transforms.py',
                  '__init__.py']
pipeline_modules = submodule_files('pipeline', pipeline_modules)
modules = (['data.py'] +
           pipeline_modules +
           ['metrics.py',
           'cv.py'])


@click.command()
@click.option('-f', '--input_file', type=click.Path(exists=True))
@click.option('-m', '--message', default="")
def cli(input_file: str, message: str = "") -> None:
    print('modules to add')
    pprint.pprint(modules)
    input_path = os.path.join(INPUT_NOTEBOOKS_DIR, input_file)
    assert os.path.exists(input_path)
    print(f'input_file = {input_path}')
    nb = replace_code(input_path, modules, message)

    os.makedirs(OUTPUT_NOTEBOOKS_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_NOTEBOOKS_DIR, input_file)
    print(f'writting output to {output_path}')
    nbformat.write(nb, output_path)


if __name__ == '__main__':
    cli()