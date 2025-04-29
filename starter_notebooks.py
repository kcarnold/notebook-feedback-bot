from pathlib import Path
import subprocess
import nbformat
from typing import Dict, List

DEFAULT_REPO_URL = "https://github.com/Calvin-Data-Science/cs375-376-public"

def update_repo(repo_dir: Path, repo_url: str = DEFAULT_REPO_URL):
    """Clone or pull the starter notebooks repository."""
    if not repo_dir.exists():
        subprocess.run(["git", "clone", repo_url, str(repo_dir)], check=True)
    else:
        subprocess.run(["git", "-C", str(repo_dir), "pull"], check=False)


def load_notebook(path: Path) -> nbformat.NotebookNode:
    """Load a notebook from a file path."""
    with open(path, 'r') as f:
        return nbformat.read(f, as_version=4)


def notebook_to_quarto(nb: nbformat.NotebookNode) -> str:
    """Convert an nbformat NotebookNode to Quarto markdown text."""
    chunks: List[str] = []
    for cell in nb.cells:
        if cell.cell_type == 'markdown':
            chunks.append(cell.source)
        elif cell.cell_type == 'code':
            chunks.append(f"```{{python}}\n{cell.source}\n```")
    return '\n\n'.join(chunks)


def get_all_starters(repo_dir: Path, repo_url: str = DEFAULT_REPO_URL) -> Dict[str, str]:
    """Return a mapping of starter notebook stems to their Quarto text."""
    update_repo(repo_dir, repo_url)
    starters_dir = repo_dir / "notebooks"
    starters: Dict[str, str] = {}
    for starter in starters_dir.glob("*.ipynb"):
        nb = load_notebook(starter)
        starters[starter.stem] = notebook_to_quarto(nb)
    return starters


def get_file_history(repo_dir: Path, path: Path) -> List[str]:
    """Return list of commit SHAs that touched the given file."""
    rel = str(path.relative_to(repo_dir))
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "log", "--pretty=format:%H", "--", rel],
        check=True, stdout=subprocess.PIPE, text=True
    )
    return out.stdout.strip().splitlines()


def get_file_content_at_rev(repo_dir: Path, path: Path, rev: str) -> nbformat.NotebookNode:
    """Load the notebook file at a specific Git revision."""
    rel = str(path.relative_to(repo_dir))
    out = subprocess.run(
        ["git", "-C", str(repo_dir), "show", f"{rev}:{rel}"],
        check=True, stdout=subprocess.PIPE, text=True
    )
    return nbformat.reads(out.stdout, as_version=4)


def get_all_starter_versions(repo_dir: Path, repo_url: str = DEFAULT_REPO_URL) -> Dict[str, Dict[str, str]]:
    """Map each starter notebook stem to a dict of revision SHA to Quarto text."""
    update_repo(repo_dir, repo_url)
    versions: Dict[str, Dict[str, str]] = {}
    starters_dir = repo_dir / "notebooks"
    for ipynb in starters_dir.glob("*.ipynb"):
        name = ipynb.stem
        versions[name] = {}
        for rev in get_file_history(repo_dir, ipynb):
            nb = get_file_content_at_rev(repo_dir, ipynb, rev)
            versions[name][rev] = notebook_to_quarto(nb)
    return versions
