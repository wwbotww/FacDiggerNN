"""Portable, directory-contained artifact locations; paths are not asset identities."""

from pathlib import Path, PurePosixPath, PureWindowsPath

from facdigger.data.contracts import DataContractError


def artifact_path(root: Path, relative: str, label: str, *, require_file: bool = True) -> Path:
    """Read POSIX or legacy Windows relative paths without accepting drives or escapes."""
    windows = PureWindowsPath(relative)
    portable = PurePosixPath(relative.replace("\\", "/"))
    if (
        not relative or windows.drive or portable.is_absolute()
        or ".." in portable.parts or portable == PurePosixPath(".")
    ):
        raise DataContractError(f"{label} escapes its directory or is not a relative artifact path")
    root = root.resolve()
    path = root.joinpath(*portable.parts).resolve()
    if not path.is_relative_to(root):
        raise DataContractError(f"{label} escapes its directory")
    if require_file and not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path
