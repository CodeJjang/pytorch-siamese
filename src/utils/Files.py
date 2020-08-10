from pathlib import Path


def create_dir_path_if_not_exist(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
