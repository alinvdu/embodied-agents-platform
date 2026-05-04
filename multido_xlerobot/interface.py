from __future__ import annotations

import importlib
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

from .bootstrap import (
    XLeRobotBootstrapError,
    XLeRobotBootstrapResult,
    bootstrap_xlerobot,
    resolve_xlerobot_repo_root,
)


class XLeRobotInterface:
    """Facade for importing and instantiating XLeRobot integrations cleanly."""

    def __init__(
        self,
        repo_root: str | Path | None = None,
        *,
        force_reload: bool = False,
    ) -> None:
        self.repo_root = resolve_xlerobot_repo_root(repo_root)
        self.force_reload = force_reload
        self._bootstrap_result: XLeRobotBootstrapResult | None = None

    @property
    def paths(self):
        return self.bootstrap().paths

    def bootstrap(self) -> XLeRobotBootstrapResult:
        if self._bootstrap_result is None or self.force_reload:
            self._bootstrap_result = bootstrap_xlerobot(
                self.repo_root,
                force_reload=self.force_reload,
            )
            self.force_reload = False
        return self._bootstrap_result

    def modules(self) -> dict[str, ModuleType]:
        result = self.bootstrap()
        return {
            "robot": result.robot_module,
            "robot_2wheels": result.robot_2wheels_module,
            "vr": result.vr_module,
            "model": result.model_module,
            "record": result.record_module,
        }

    def robot_classes(self) -> tuple[type[Any], type[Any]]:
        robot_module = self.bootstrap().robot_module
        return robot_module.XLerobotConfig, robot_module.XLerobot

    def robot_2wheels_classes(self) -> tuple[type[Any], type[Any]]:
        module = self.bootstrap().robot_2wheels_module
        if module is None:
            raise XLeRobotBootstrapError("xlerobot_2wheels module is unavailable in this environment")
        config_cls = getattr(module, "XLerobot2WheelsConfig", None)
        if config_cls is None:
            config_module = importlib.import_module(
                f"{module.__name__}.config_xlerobot_2wheels"
            )
            config_cls = config_module.XLerobot2WheelsConfig
        return config_cls, module.XLerobot2Wheels

    def vr_classes(self) -> tuple[type[Any], type[Any]]:
        vr_module = self.bootstrap().vr_module
        return vr_module.XLerobotVRTeleopConfig, vr_module.XLerobotVRTeleop

    def model_classes(self) -> dict[str, Any]:
        module = self.bootstrap().model_module
        return {
            name: getattr(module, name)
            for name in dir(module)
            if not name.startswith("_")
        }

    def record_module(self) -> ModuleType:
        module = self.bootstrap().record_module
        if module is None:
            raise XLeRobotBootstrapError("record module is unavailable in this environment")
        return module

    def make_robot_config(self, **overrides: Any) -> Any:
        config_cls, _ = self.robot_classes()
        return config_cls(**overrides)

    def make_robot(self, **config_overrides: Any) -> Any:
        config_cls, robot_cls = self.robot_classes()
        return robot_cls(config_cls(**config_overrides))

    def make_2wheels_robot_config(self, **overrides: Any) -> Any:
        config_cls, _ = self.robot_2wheels_classes()
        return config_cls(**overrides)

    def make_2wheels_robot(self, **config_overrides: Any) -> Any:
        config_cls, robot_cls = self.robot_2wheels_classes()
        return robot_cls(config_cls(**config_overrides))

    def make_vr_config(self, **overrides: Any) -> Any:
        config_cls, _ = self.vr_classes()
        self._prepare_vr_overrides(overrides)
        return config_cls(**overrides)

    def make_vr_teleop(self, **config_overrides: Any) -> Any:
        config_cls, teleop_cls = self.vr_classes()
        self._prepare_vr_overrides(config_overrides)
        return teleop_cls(config_cls(**config_overrides))

    def _prepare_vr_overrides(self, overrides: dict[str, Any]) -> None:
        xlevr_path = Path(overrides.get("xlevr_path") or self.paths.xlevr_root).expanduser().resolve()
        overrides["xlevr_path"] = str(xlevr_path)
        _patch_imported_vr_monitor_path(self.bootstrap().vr_module.__name__, xlevr_path)

    def summary(self) -> dict[str, str]:
        result = self.bootstrap()
        return {
            "repo_root": str(result.paths.repo_root),
            "software_src": str(result.paths.software_src),
            "xlevr_root": str(result.paths.xlevr_root),
            "record_script": str(result.paths.record_script),
            "robot_module": result.robot_module.__name__,
            "robot_2wheels_module": result.robot_2wheels_module.__name__
            if result.robot_2wheels_module is not None
            else "unavailable",
            "vr_module": result.vr_module.__name__,
            "model_module": result.model_module.__name__,
            "record_module": result.record_module.__name__ if result.record_module is not None else "unavailable",
        }

    @staticmethod
    def installation_help() -> str:
        return (
            "XLeRobot extends LeRobot rather than replacing it. Use a Python "
            "environment where `lerobot` is installed, then point this adapter to "
            "your XLeRobot fork with XLEROBOT_FORKED_ROOT or repo_root=..."
        )


def _patch_imported_vr_monitor_path(vr_module_name: str, xlevr_path: str | Path) -> bool:
    """Point the fork's imported VR monitor module at the selected XLeVR checkout."""
    monitor_module = sys.modules.get(f"{vr_module_name}.vr_monitor")
    if monitor_module is None or not hasattr(monitor_module, "XLEVR_PATH"):
        return False
    monitor_module.XLEVR_PATH = str(Path(xlevr_path).expanduser().resolve())
    return True
