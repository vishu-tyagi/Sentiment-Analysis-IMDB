import os
import glob
import logging
import logging.config
from pathlib import Path

import tarfile
import requests

from sentiment_analysis.utils import timing

logger = logging.getLogger(__name__)


@timing
def download_data(url, save_to: Path) -> None:
    """
    Download data from URL

    Args:
        data_url (_type_): URL to download from
        to_ (Path): Destination for downloaded data
    """
    logger.info(f"Downloading {url}")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(str(save_to), 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)


@timing
def unzip_tar(file_path: Path, keep_dirs: list[str], extract_to: Path) -> None:
    """
    Unzip a tar file

    Args:
        file_path (Path): Path to the tar file
        keep_dirs (list[str]): Directories to extract from the tar file
        extract_to (Path): Path for the extracted directories

    Raises:
        Exception: When file_path does not point to a tar file
        ValueError: When file_path does not point to a tar file
    """
    try:
        if str(file_path).endswith("tar.gz"):
            tar = tarfile.open(str(file_path), "r:gz")
        elif str(file_path).endswith("tar"):
            tar = tarfile.open(str(file_path), "r:")
        else:
            raise Exception
    except Exception:
        raise ValueError("Not a valid tar file")
    logger.info("Following folders will be retained:")
    for i, f in enumerate(keep_dirs):
        logger.info(f"{i+1}. {f}")
    members = [f for f in tar.getmembers() if f.name.startswith(tuple(keep_dirs))]
    tar.extractall(members=members, path=extract_to)
    logger.info("Files unpacked")
    tar.close()


def collect_files(dirs: list[Path], file_types: list[str]) -> list[Path]:
    """
    Return relevant files from a list of directories

    Args:
        dirs (list[Path]): List of directories to fetch files from
        file_types (list[str]): Type of files to fetch, for example,
        `["*.txt"]` is a common value for this parameter.

    Returns:
        list[Path]: List of paths that point to files found
    """
    all_files = []
    for dir in dirs:
        for file_type in file_types:
            files = glob.glob(os.path.join(dir, file_type))
            all_files += files
    return all_files
