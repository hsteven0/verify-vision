from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
WINDOWS_LAUNCHER = REPOSITORY_ROOT / "verifyvision.ps1"
LINUX_LAUNCHER = REPOSITORY_ROOT / "verifyvision.sh"
WINDOWS_EXE = REPOSITORY_ROOT / "VerifyVision.exe"
WINDOWS_EXE_SOURCE = REPOSITORY_ROOT / "scripts" / "windows-launcher" / "Launcher.cs"
WINDOWS_EXE_BUILD = REPOSITORY_ROOT / "scripts" / "windows-launcher" / "build.ps1"
WINDOWS_EXE_ICON = REPOSITORY_ROOT / "scripts" / "windows-launcher" / "VerifyVision.ico"
VERSION = (REPOSITORY_ROOT / "backend" / "app" / "VERSION").read_text().strip()


def powershell() -> str | None:
    return shutil.which("pwsh") or shutil.which("powershell")


def run_windows_launcher(*arguments: str, cwd: Path, test_mode: bool = False) -> subprocess.CompletedProcess[str]:
    executable = powershell()
    if executable is None:
        pytest.skip("PowerShell is not available")
    environment = os.environ.copy()
    if test_mode:
        environment["VERIFYVISION_LAUNCHER_TEST_MODE"] = "1"
    return subprocess.run(
        [
            executable,
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(WINDOWS_LAUNCHER),
            *arguments,
        ],
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher test")
def test_windows_launcher_help_and_version_are_path_independent(tmp_path: Path) -> None:
    help_result = run_windows_launcher(cwd=tmp_path)
    version_result = run_windows_launcher("version", cwd=tmp_path)

    assert help_result.returncode == 0
    assert ".\\verifyvision.ps1 <command>" in help_result.stdout
    assert version_result.returncode == 0
    assert version_result.stdout.strip() == f"VerifyVision v{VERSION}"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher test")
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("setup", "dispatch:setup:scripts/setup-local.ps1"),
        ("start", "dispatch:start:scripts/manage-local.ps1"),
        ("stop", "dispatch:stop:scripts/manage-local.ps1"),
        ("status", "dispatch:status:scripts/manage-local.ps1"),
        ("diagnose", "dispatch:diagnose:scripts/manage-local.ps1"),
        ("update", "dispatch:update:scripts/update-dependencies.ps1"),
        ("all", "dispatch:all:setup-if-needed,start"),
    ],
)
def test_windows_launcher_dispatches_without_running_installers(command: str, expected: str, tmp_path: Path) -> None:
    result = run_windows_launcher(command, cwd=tmp_path, test_mode=True)

    assert result.returncode == 0
    assert expected in result.stdout


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher test")
def test_windows_launcher_rejects_unknown_command(tmp_path: Path) -> None:
    result = run_windows_launcher("unknown", cwd=tmp_path)

    assert result.returncode == 2
    assert "Unknown command" in result.stderr


def test_launchers_use_the_canonical_version_file() -> None:
    assert "backend\\app\\VERSION" in WINDOWS_LAUNCHER.read_text()
    assert "backend/app/VERSION" in LINUX_LAUNCHER.read_text()
    assert f"VerifyVision v{VERSION}" not in WINDOWS_LAUNCHER.read_text()
    assert f"VerifyVision v{VERSION}" not in LINUX_LAUNCHER.read_text()


def test_linux_launcher_has_expected_dispatch_and_fixed_ports() -> None:
    launcher = LINUX_LAUNCHER.read_text()
    manager = (REPOSITORY_ROOT / "scripts" / "manage-local.sh").read_text()

    for command in ("all", "setup", "start", "stop", "status", "diagnose", "update"):
        assert command in launcher
    assert "BACKEND_PORT=8000" in manager
    assert "FRONTEND_PORT=5173" in manager
    assert "VERIFYVISION_LAUNCHER_TEST_MODE" in launcher


def test_start_skips_full_diagnostics_and_launches_both_services_first() -> None:
    windows = (REPOSITORY_ROOT / "scripts" / "manage-local.ps1").read_text()
    linux = (REPOSITORY_ROOT / "scripts" / "manage-local.sh").read_text()

    assert "$diagnosticLog" not in windows
    assert "diagnostic_log=" not in linux
    assert windows.index("$frontend = Start-Process") < windows.index("$health = Wait-ForBackend")
    assert linux.index('nohup node "$VITE_SCRIPT"') < linux.index('wait_for_url "$backend_pid"')


def test_windows_exe_launcher_uses_repo_relative_paths_and_existing_health_checks() -> None:
    source = WINDOWS_EXE_SOURCE.read_text()

    assert "AppDomain.CurrentDomain.BaseDirectory" in source
    assert 'Path.Combine(root, "verifyvision.ps1")' in source
    assert "B:\\Projects\\VerifyVision" not in source
    assert "http://127.0.0.1:8000/api/health" in source
    assert source.index("if (!IsReady())") < source.index("RunPowerShell(root, script")
    assert "-NoProfile -NonInteractive -ExecutionPolicy Bypass -File" in source
    assert "CreateNoWindow = true" in source
    assert "UseShellExecute = true" in source
    assert "private static int Main()" in source
    assert "VerifyVision is not set up yet." in source
    assert r".\\verifyvision.ps1 diagnose" in source


def test_windows_exe_build_uses_release_version_and_multisize_icon() -> None:
    build = WINDOWS_EXE_BUILD.read_text()
    icon = WINDOWS_EXE_ICON.read_bytes()

    reserved, image_type, image_count = struct.unpack_from("<HHH", icon)
    sizes = {
        (
            256 if icon[6 + index * 16] == 0 else icon[6 + index * 16],
            256 if icon[7 + index * 16] == 0 else icon[7 + index * 16],
        )
        for index in range(image_count)
    }

    assert "backend\\app\\VERSION" in build
    assert "/win32icon" in build
    assert reserved == 0
    assert image_type == 1
    assert sizes == {(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)}


@pytest.mark.skipif(sys.platform != "win32", reason="Windows launcher test")
def test_windows_exe_has_release_metadata() -> None:
    executable = powershell()
    if executable is None:
        pytest.skip("PowerShell is not available")
    command = (
        "$v=(Get-Item -LiteralPath '"
        + str(WINDOWS_EXE)
        + "').VersionInfo; "
        + "[pscustomobject]@{ProductName=$v.ProductName;FileDescription=$v.FileDescription;"
        + "ProductVersion=$v.ProductVersion;FileVersion=$v.FileVersion;"
        + "OriginalFilename=$v.OriginalFilename} | ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        [executable, "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )

    assert json.loads(result.stdout) == {
        "ProductName": "VerifyVision",
        "FileDescription": "VerifyVision",
        "ProductVersion": VERSION,
        "FileVersion": f"{VERSION}.0",
        "OriginalFilename": "VerifyVision.exe",
    }


def test_windows_exe_is_a_gui_application() -> None:
    executable = WINDOWS_EXE.read_bytes()
    pe_offset = struct.unpack_from("<I", executable, 0x3C)[0]
    subsystem = struct.unpack_from("<H", executable, pe_offset + 24 + 68)[0]

    assert executable[:2] == b"MZ"
    assert executable[pe_offset : pe_offset + 4] == b"PE\0\0"
    assert subsystem == 2
