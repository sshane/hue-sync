#!/usr/bin/env python3
"""
pykit_auth_test.py

End-to-end "pure hue_entertainment_pykit" test:
- waits for bridge button press
- registers (POST /api, generateclientkey=true)
- fetches bridge hue-application-id (GET /auth/v1)
- fetches entertainment configurations (CLIP v2)
- starts the selected entertainment configuration and performs DTLS handshake

Run:
  python3 ./pykit_auth_test.py --bridge-ip <BRIDGE_IP> --ent-id <ENT_ID_UUID> --timeout 30

Security note:
- This script intentionally avoids writing `data/auth.json` into the repo by running pykit in a temporary directory.
- Do not print or commit real Hue keys/clientkeys.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from typing import Optional

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hue Entertainment pykit auth + DTLS handshake test")
    p.add_argument("--bridge-ip", required=True)
    p.add_argument("--ent-id", required=True, help="Entertainment configuration UUID (clip/v2 id)")
    p.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for link button (default: 30)")
    return p.parse_args()


def _enter_temp_pykit_dir() -> str:
    """
    hue_entertainment_pykit persists auth to ./data/auth.json via its FileHandler (cwd-based).
    For a public repo, we do NOT want it writing credential files into the project directory.

    This creates a temp dir, chdirs into it, and creates ./data/ so pykit can write there.
    Returns the temp dir path.
    """
    d = tempfile.mkdtemp(prefix="hue_pykit_")
    os.chdir(d)
    os.makedirs("data", exist_ok=True)
    return d


def wait_for_button_and_fetch_bridge_data(bridge_ip: str, timeout_s: float) -> dict:
    # Import lazily (after cwd is set) so its FileHandler writes into this project.
    from bridge.bridge_repository import BridgeRepository
    from exceptions.bridge_exception import BridgeException

    repo = BridgeRepository()

    deadline = time.time() + timeout_s
    last_err: Optional[str] = None
    while time.time() < deadline:
        try:
            return repo.fetch_bridge_data(bridge_ip)
        except BridgeException as e:
            last_err = str(e)
            # most common error is "link button not pressed"
            time.sleep(1.0)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(1.0)

    raise SystemExit(f"Timed out waiting for link button ({timeout_s:.0f}s). Last error: {last_err}")


def main() -> None:
    args = parse_args()
    tmp = _enter_temp_pykit_dir()

    # Import the pykit module first so its sys.path/sys.modules conflict workaround
    # takes effect (common collision: a different top-level `services` package).
    import hue_entertainment_pykit  # noqa: F401

    print(f"pykit auth test: press the Hue Bridge button now (waiting {args.timeout:.0f}s)…")
    bridge_data = wait_for_button_and_fetch_bridge_data(args.bridge_ip, args.timeout)
    print("bridge_data keys:", sorted(bridge_data.keys()))
    print("username:", bridge_data.get("username"))
    print("hue-application-id:", bridge_data.get("hue-application-id"))
    # Do NOT print the raw clientkey; treat it as a secret.
    print("clientkey: <redacted>")
    print(f"(pykit wrote credentials under temporary dir: {tmp})")

    # Build Bridge + repositories
    from models.bridge import Bridge
    from bridge.entertainment_configuration_repository import EntertainmentConfigurationRepository
    from services.streaming_service import StreamingService
    from network.dtls import Dtls

    bridge = Bridge.from_dict(bridge_data)
    ent_repo = EntertainmentConfigurationRepository(bridge)
    configs = ent_repo.fetch_configurations()
    if args.ent_id not in configs:
        print("Available entertainment_configuration ids:")
        for k, cfg in configs.items():
            print(" -", k, getattr(cfg, "name", None), getattr(cfg, "configuration_type", None))
        raise SystemExit(f"--ent-id {args.ent_id!r} not found on bridge")

    cfg = configs[args.ent_id]
    print("Selected entertainment config:", cfg.id, "name=", getattr(cfg, "name", None), "status=", getattr(cfg, "status", None))

    dtls = Dtls(bridge)
    streaming = StreamingService(cfg, ent_repo, dtls)

    # Start stream triggers: PUT action=start then DTLS handshake
    print("Starting stream (this will attempt DTLS handshake)…")
    try:
        streaming.start_stream()
    except Exception as e:
        print("START_STREAM FAILED:", type(e).__name__, e)
        raise

    print("START_STREAM OK (DTLS handshake succeeded). Keeping alive for 3 seconds…")
    time.sleep(3.0)

    try:
        streaming.stop_stream()
        print("STOP_STREAM OK")
    except Exception as e:
        print("STOP_STREAM FAILED:", type(e).__name__, e)


if __name__ == "__main__":
    main()


