import importlib.util
import pathlib
import sys
import subprocess
import types
import unittest
from unittest.mock import patch


_PACKAGE_DIR = pathlib.Path(__file__).parents[1] / "python" / "infinilm"
_infinilm = types.ModuleType("infinilm")
_infinilm.__path__ = [str(_PACKAGE_DIR)]
sys.modules.setdefault("infinilm", _infinilm)
_moe_config = types.ModuleType("infinilm.moe_config")
_moe_config.MOE_EP_BACKEND_HELP = ""
sys.modules.setdefault("infinilm.moe_config", _moe_config)

_spec = importlib.util.spec_from_file_location(
    "infinilm.base_config", _PACKAGE_DIR / "base_config.py"
)
_base_config = importlib.util.module_from_spec(_spec)
sys.modules["infinilm.base_config"] = _base_config
_spec.loader.exec_module(_base_config)
_infinilm.base_config = _base_config
BaseConfig = _base_config.BaseConfig


class BaseConfigDeviceDetectionTest(unittest.TestCase):
    @staticmethod
    def only_dlsmi(command):
        return "/usr/bin/dlsmi" if command == "dlsmi" else None

    def test_detects_qy_from_successful_dlsmi_probe(self):
        config = BaseConfig.__new__(BaseConfig)

        with patch(
            "infinilm.base_config.shutil.which", return_value="/usr/bin/dlsmi"
        ), patch("infinilm.base_config.subprocess.run") as run, patch.object(
            config, "_torch_device_available", return_value=False
        ):
            self.assertEqual(config.detect_device(), "qy")

        run.assert_called_once_with(
            ["/usr/bin/dlsmi"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=2,
        )

    def test_ignores_failed_dlsmi_probe(self):
        config = BaseConfig.__new__(BaseConfig)

        with patch(
            "infinilm.base_config.shutil.which",
            side_effect=self.only_dlsmi,
        ), patch(
            "infinilm.base_config.subprocess.run",
            side_effect=subprocess.CalledProcessError(1, "dlsmi"),
        ), patch.object(config, "_torch_device_available", return_value=False):
            self.assertEqual(config.detect_device(), "cpu")


if __name__ == "__main__":
    unittest.main()
