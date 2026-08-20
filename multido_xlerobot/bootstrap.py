# Copyright 2026 Alin Vasile Dumitru
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .types import XLeRobotPaths


def _default_repo_root() -> Path:
    candidates = [
        Path.home() / "xlerobot_forked",
        Path.home() / "XLeRobot",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


DEFAULT_XLEROBOT_FORK_ROOT = _default_repo_root()


def resolve_xlerobot_repo_root(repo_root: str | Path | None = None) -> Path:
    root = repo_root or os.environ.get("XLEROBOT_FORKED_ROOT") or DEFAULT_XLEROBOT_FORK_ROOT
    return Path(root).expanduser().resolve()


class XLeRobotBootstrapError(RuntimeError):
    """Raised when the local XLeRobot fork cannot be exposed cleanly."""


@dataclass(frozen=True)
class XLeRobotBootstrapResult:
    paths: XLeRobotPaths
    robot_module: ModuleType
    robot_2wheels_module: ModuleType | None
    vr_module: ModuleType
    model_module: ModuleType
    record_module: ModuleType | None


def bootstrap_xlerobot(
    repo_root: str | Path | None = None,
    *,
    force_reload: bool = False,
) -> XLeRobotBootstrapResult:
    paths = XLeRobotPaths.from_repo_root(resolve_xlerobot_repo_root(repo_root))
    _validate_paths(paths)
    _ensure_lerobot_installed()
    _install_compatibility_aliases()

    model_module = _load_module(
        "lerobot.model.SO101Robot",
        paths.model_file,
        force_reload=force_reload,
    )
    robot_module = _load_package(
        "lerobot.robots.xlerobot",
        paths.robot_pkg_dir,
        force_reload=force_reload,
    )
    robot_2wheels_module = _load_optional_package(
        "lerobot.robots.xlerobot_2wheels",
        paths.robot_2wheels_pkg_dir,
        force_reload=force_reload,
    )
    vr_module = _load_package(
        "lerobot.teleoperators.xlerobot_vr",
        paths.vr_pkg_dir,
        force_reload=force_reload,
    )
    record_module = _load_optional_module(
        "xlerobot_forked.record",
        paths.record_script,
        force_reload=force_reload,
    )

    return XLeRobotBootstrapResult(
        paths=paths,
        robot_module=robot_module,
        robot_2wheels_module=robot_2wheels_module,
        vr_module=vr_module,
        model_module=model_module,
        record_module=record_module,
    )


def _validate_paths(paths: XLeRobotPaths) -> None:
    required = {
        "repo_root": paths.repo_root,
        "software_src": paths.software_src,
        "record_script": paths.record_script,
        "model_file": paths.model_file,
        "robot_pkg_dir": paths.robot_pkg_dir,
        "robot_2wheels_pkg_dir": paths.robot_2wheels_pkg_dir,
        "vr_pkg_dir": paths.vr_pkg_dir,
    }
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        formatted = ", ".join(f"{name}={required[name]}" for name in missing)
        raise XLeRobotBootstrapError(
            f"XLeRobot fork is incomplete or moved. Missing: {formatted}"
        )


def _ensure_lerobot_installed() -> None:
    try:
        importlib.import_module("lerobot")
    except ModuleNotFoundError as exc:
        raise XLeRobotBootstrapError(
            "The XLeRobot fork extends the `lerobot` package, but `lerobot` is not "
            "installed in the current Python environment. Install a compatible "
            "LeRobot environment first, then bootstrap XLeRobot."
        ) from exc


def _install_compatibility_aliases() -> None:
    try:
        modern_module = importlib.import_module("lerobot.robots.so101_follower.config_so101_follower")
    except ModuleNotFoundError:
        return

    legacy_pkg_name = "lerobot.robots.so_follower"
    legacy_module_name = f"{legacy_pkg_name}.config_so_follower"
    if legacy_module_name in sys.modules:
        return

    legacy_pkg = ModuleType(legacy_pkg_name)
    legacy_pkg.__path__ = []
    setattr(legacy_pkg, "config_so_follower", modern_module)
    sys.modules[legacy_pkg_name] = legacy_pkg
    sys.modules[legacy_module_name] = modern_module


def _load_package(module_name: str, package_dir: Path, *, force_reload: bool) -> ModuleType:
    init_file = package_dir / "__init__.py"
    if not init_file.exists():
        raise XLeRobotBootstrapError(f"Missing package entrypoint: {init_file}")

    _ensure_parent_packages(module_name)
    if force_reload:
        _purge_modules(module_name)
    elif module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(
        module_name,
        init_file,
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise XLeRobotBootstrapError(f"Unable to build import spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _attach_to_parent(module_name, module)
    return module


def _load_optional_package(
    module_name: str,
    package_dir: Path,
    *,
    force_reload: bool,
) -> ModuleType | None:
    try:
        return _load_package(module_name, package_dir, force_reload=force_reload)
    except Exception:
        return None


def _load_module(module_name: str, file_path: Path, *, force_reload: bool) -> ModuleType:
    if not file_path.exists():
        raise XLeRobotBootstrapError(f"Missing module file: {file_path}")

    _ensure_parent_packages(module_name)
    if force_reload and module_name in sys.modules:
        del sys.modules[module_name]
    elif module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise XLeRobotBootstrapError(f"Unable to build import spec for {module_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _attach_to_parent(module_name, module)
    return module


def _load_optional_module(
    module_name: str,
    file_path: Path,
    *,
    force_reload: bool,
) -> ModuleType | None:
    try:
        return _load_module(module_name, file_path, force_reload=force_reload)
    except Exception:
        return None


def _purge_modules(prefix: str) -> None:
    names = [name for name in sys.modules if name == prefix or name.startswith(f"{prefix}.")]
    for name in names:
        del sys.modules[name]


def _ensure_parent_packages(module_name: str) -> None:
    parts = module_name.split(".")
    for index in range(1, len(parts)):
        package_name = ".".join(parts[:index])
        if package_name in sys.modules:
            continue
        try:
            importlib.import_module(package_name)
            continue
        except ModuleNotFoundError:
            pass

        package = ModuleType(package_name)
        package.__path__ = []
        sys.modules[package_name] = package


def _attach_to_parent(module_name: str, module: ModuleType) -> None:
    if "." not in module_name:
        return
    parent_name, child_name = module_name.rsplit(".", 1)
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_name, module)
