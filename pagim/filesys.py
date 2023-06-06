import os
import shutil
from typing import Iterator


def reset_dir(path: str) -> None:
    """Remove the entire dir and recreate it"""
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path, exist_ok=True)


def walkdir_with_extension(path: str, extensions: list = None, level: int = None) -> Iterator[str]:
    """Walk a directory recursively and find files of certain extension

    Args:
        path: The root directory to walk
        extensions: The list of extensions that files should match. Empty list
                    to return all files.
        level: Number of recursive levels to walk, 0 to yield only files under
               the specified path

    Yields:
        path to the files that carries the (case-insensitive) extension.
    """
    # sanitize
    if extensions:
        extensions = [str(x).lower() for x in extensions]
    # walk dir in depth-first search
    basecount = path.count("/")
    for root, dirs, files in os.walk(path):
        if level is not None and root.count("/") > basecount+level:
            # skip, but can't break yet due to depth-first search
            continue
        for filename in files:
            if extensions:
                # note: filename..jpg -> no ext due to >1 dots
                _, ext = os.path.splitext(filename)
                if ext.lower() not in extensions:
                    continue
            yield os.path.join(root, filename)


def norm_path(path: str) -> str:
    """Get the symlink-resolved, normalized path"""
    return os.path.realpath(os.path.abspath(os.path.expanduser(path)))


def project_root() -> str:
    """Return the git project dir"""
    return os.path.normpath(os.path.join(os.path.dirname(norm_path(__file__)), ".."))
