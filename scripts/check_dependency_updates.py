from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = REPOSITORY_ROOT / "backend"
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.dependencies.service import DependencyUpdateService


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check for dependency updates without installing them."
    )
    parser.add_argument(
        "--cached",
        action="store_true",
        help="Show the last cached result instead of contacting package registries.",
    )
    args = parser.parse_args()

    service = DependencyUpdateService(
        REPOSITORY_ROOT / ".cache" / "dependency-update-status.json"
    )
    status = service.status() if args.cached else service.check_now()

    print(f"Status: {status.state.replace('_', ' ')}")
    if status.last_checked_at is not None:
        print(f"Last checked: {status.last_checked_at.isoformat()}")
    if status.detail:
        print(status.detail)

    for update in status.updates:
        risk = "ML runtime" if update.risk == "critical_runtime" else "Application"
        print(
            f"{update.package:<32} {update.current_version:<18} -> {update.latest_version:<18} {risk}"
        )

    print("No packages were changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
