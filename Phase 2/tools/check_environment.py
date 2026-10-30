"""Compare installed packages with exact pins in requirements.txt."""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
req = root / "requirements.txt"
failed = False
for line in req.read_text().splitlines():
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    match = re.match(r"([A-Za-z0-9_.-]+)==([^;\s]+)", line)
    if not match:
        continue
    package, expected = match.groups()
    try:
        actual = version(package)
    except PackageNotFoundError:
        actual = "NOT INSTALLED"
    status = "OK" if actual == expected else "MISMATCH"
    print(f"{status:10} {package:28} expected={expected:12} installed={actual}")
    failed |= status != "OK"
raise SystemExit(1 if failed else 0)
