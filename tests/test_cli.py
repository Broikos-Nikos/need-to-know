import subprocess
import sys
from importlib.metadata import entry_points

import pytest

from needtoknow import __version__
from needtoknow.cli import main


def test_version_flag_prints_the_package_version(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(["--version"])
    assert exit_info.value.code == 0
    assert capsys.readouterr().out.strip() == f"ntk {__version__}"


def test_no_arguments_prints_help_and_exits_zero(capsys):
    assert main([]) == 0
    assert "usage: ntk" in capsys.readouterr().out


def test_the_ntk_console_script_points_at_main():
    # Reads the installed metadata, so a wrong [project.scripts] line fails here.
    (script,) = entry_points(group="console_scripts", name="ntk")
    assert script.load() is main


def test_the_module_runs_as_a_program():
    result = subprocess.run([sys.executable, "-m", "needtoknow.cli", "--version"], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == f"ntk {__version__}"
