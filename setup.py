"""Packaging metadata for MediX Agent Swarm.

The project still ships as an alpha/prototype.  Runtime data is included in a
platform-neutral ``share/medix-agent-swarm`` directory; source-checkout mode
continues to discover the same files from the repository.
"""

from pathlib import Path
from typing import List, Tuple

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent


def read_requirements() -> List[str]:
    requirements: List[str] = []
    for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)
    return requirements


def runtime_data_files() -> List[Tuple[str, List[str]]]:
    """Collect non-secret Skill definitions and seed medical documents."""

    groups: dict[str, List[str]] = {}
    for base in (ROOT / ".claude" / "skills", ROOT / "knowledge" / "data" / "documents"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            relative_parent = path.parent.relative_to(ROOT)
            target = str(Path("share") / "medix-agent-swarm" / relative_parent)
            groups.setdefault(target, []).append(str(path))
    return sorted(groups.items())


setup(
    name="medix-agent-swarm",
    version="0.2.0",
    author="MediX Team",
    description="Prototype multi-agent medical information assistant",
    long_description=(ROOT / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("examples", "examples.*")),
    py_modules=["main"],
    include_package_data=True,
    package_data={"constraints": ["*.yaml"]},
    data_files=runtime_data_files(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Healthcare Industry",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.10",
    install_requires=read_requirements(),
    entry_points={"console_scripts": ["medix-assistant=main:main"]},
)
