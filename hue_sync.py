#!/usr/bin/env python3
# hue_sync.py
#
# EPIC Hue Entertainment Audio Sync (Linux)
# - Plays an audio file and streams Hue Entertainment frames at 50Hz (default)
# - DTLS PSK over UDP/2100 (Entertainment API)
#
# Why this is fast:
# - ONE UDP DTLS connection
# - ONE binary packet per frame that contains all channels
# - No per-light REST PUT loops
#
# Entertainment protocol basics (HueStream header, UUID, channel RGB16 data):
# See: https://iotech.blog/posts/philips-hue-entertainment-api/
#
# Dependencies:
#   pip install numpy sounddevice soundfile requests python-mbedtls
#
# Example usage:
#   ./hue_sync.py \
#     --bridge-ip <BRIDGE_IP> \
#     --app-key "<USERNAME>" \
#     --clientkey-hex "<CLIENTKEY_HEX>" \
#     --ent-id "<ENTERTAINMENT_UUID>" \
#     --file song.mp3
#
# Notes:
# - clientkey from Hue registration is typically returned as a hex string.
# - You must have an Entertainment Area created in the Hue app.
# - This script starts the entertainment area via HTTPS CLIP v2 (minimal, only for start/stop).
#   If you truly want zero HTTPS, start the entertainment area manually (but it usually stops quickly without stream).
#
# If python-mbedtls DTLS import fails, you likely don't have the right package version.
# You want: "python-mbedtls" by Synss (mbed TLS bindings).

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import select
import sys
import termios
import tty
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Audio
import sounddevice as sd
import soundfile as sf

# Minimal HTTPS just to start/stop the entertainment area (not used for per-light updates)
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# -----------------------------
# Quick Show Toggles (edit these)
# -----------------------------

# -----------------------------
# Single-source defaults (CLI + runtime)
# -----------------------------

DEFAULT_PREVIEW_HZ = 50.0
DEFAULT_PREVIEW_SCALE = 165
DEFAULT_PREVIEW_LAYOUT = "grid"
DEFAULT_SPECTRUM_HEIGHT = 320
DEFAULT_SPECTRUM_MAX_HZ = 20000.0
DEFAULT_SPECTRUM_FFT_N = 16384

class ShowOptions:
    """
    Simple feature toggles so you can quickly enable/disable parts of the show.
    Edit these constants and restart the script.
    """

    ENABLE_KICKS = True          # low kick band group + kick-driven flashes (with subset selection)
    ENABLE_HIGH = True           # "high kick" band group (kick click + hats combined, with subset selection)
    ENABLE_BACKGROUND = True     # dim background on non-kick lights
    ENABLE_PALETTES = True       # use palettes; if False -> fixed warm hue
    ENABLE_PALETTE_SWITCH = True # allow palette changes on double-time / periodic kicks

    # Gains (tuning knobs)
    HIGH_GAIN = 2.0              # extra gain for high-band group intensity (5k–15k is often subtle)
    COLOR_VIBRANCE = 1.35        # multiplies saturation globally (>=1 = less pastel)

    # High-band (hats/clicks) response shaping:
    # The high band is naturally spiky; we slow its visual envelope so it doesn't slam to max instantly.
    HIGH_ATTACK_RATE = 4.0       # units/sec slew limit on rising edge (lower = slower ramp up)
    HIGH_RELEASE_RATE = 1.2      # units/sec decay rate for the high envelope (lower = longer tail)
    HIGH_LP_RC = 0.07            # seconds RC for high-band low-pass filter (higher = smoother/slower)

    # Extra "show" moments (iLightShow-ish)
    # These can feel "random" if the beat tracker isn't fully locked yet, so default them OFF.
    ENABLE_BUILDUP_EFFECTS = False
    ENABLE_DROP_FLASH = False
    ENABLE_WHITE_SPARKLES = False

    BUILDUP_RISER_THR = 0.55       # riser threshold to start buildup mode
    BUILDUP_HOLD_S = 1.4           # how long buildup mode persists once triggered
    BUILDUP_SPARKLES_PER_S = 7.0   # base sparkles/sec during buildup
    BUILDUP_SPARKLE_DECAY_S = 0.12 # sparkle decay time
    BUILDUP_WHITE_MAX = 0.95       # max whiteness mix during buildup

    DROP_RISER_PEAK_THR = 0.72     # if riser was above this, a subsequent fall can trigger a drop flash
    DROP_RISER_FALL_THR = 0.33
    DROP_FLASH_DECAY_S = 0.25
    DROP_WHITE = 0.85              # whiteness mix on drop
    DROP_PALETTE_JUMP = True       # palette jump on drop moments
    DROP_MIN_BEAT_ACTIVE = True    # only allow drop flashes when beat is confident/active

    # Motion / variety
    ENABLE_HUE_DRIFT = True
    DRIFT_RATE = 0.06              # hue drift cycles/sec (slow global motion)
    BUILDUP_DRIFT_BOOST = 2.6      # drift multiplier during buildup

    # Beat animation (iLightShow-ish):
    # Instead of "whatever the analyzer pulse happens to be this frame", trigger a short brightness
    # animation on each beat and scale its duration with BPM. This makes beats feel intentional
    # and prevents tiny/short detections from only hitting ~20% brightness.
    BEAT_ANIM_ENABLED = True
    BEAT_ANIM_MIN_PEAK = 0.65    # minimum added brightness on a beat (0..1). Raised to avoid weak flashes.
    BEAT_ANIM_MAX_PEAK = 1.00    # maximum added brightness on a beat (0..1)
    BEAT_ANIM_DECAY_FRAC = 0.55  # decay time as fraction of beat period (typ ~0.4..0.7)
    BEAT_ANIM_DECAY_MIN_S = 0.10
    BEAT_ANIM_DECAY_MAX_S = 0.55
    BEAT_USE_TEMPO_GRID = True   # if True and bpm_conf is high, drive beat triggers from f.beat (tempo grid) vs f.kick
    BEAT_ANIM_RETRIGGER_MIN_FRAC = 0.35  # ignore extra triggers inside this fraction of a beat (prevents chatter)
    BEAT_ANIM_RETRIGGER_MIN_S = 0.09     # absolute floor in seconds

    # Color/palette cadence (reduces "flashy"):
    COLOR_ADVANCE_EVERY_BEATS = 4  # advance palette color index every N beats (or kicks if no tempo)
    GROUP_RESELECT_EVERY_BEATS = 8 # reselect kick/high light subsets every N beats (or kicks if no tempo)

    # Frequency bands (Hz) used for driving the show + spectrum markers.
    # These are intentionally *not* CLI flags to keep tuning centralized.
    # EDM kicks often have a sub thump + punch; a slightly wider band helps on bass-heavy mixes.
    KICK_BAND = (35.0, 180.0)    # low kick / bass punch
    # Wider high band helps EDM transients (hats/claps/snares often extend well past 8kHz).
    HIGH_BAND = (3500.0, 12000.0) # kick click + hats / transient energy

    # Beat/BPM tracking tuning
    BEAT_CONF_MIN = 1.55          # bpm_conf needed to consider tempo "active"
    ONSET_EVT_STD_K = 2.2         # onset event threshold = mean + K*std
    ONSET_EVT_FLOOR = 0.010       # absolute onset threshold floor
    ONSET_EVT_REFRACTORY_S = 0.085 # minimum time between onset events used for phase lock
    ONSET_FLUX_GAIN = 6.0         # scales spectral-flux deviation when used as an onset source (normalization)

    # Quiet-section gating (prevents noisy "beats" when audio is near-silent)
    BEAT_MIN_RMS = 0.0012         # suppress onset events/phase-lock below this
    BEAT_STATS_MIN_RMS = 0.0010   # don't update onset mean/var below this (prevents threshold collapse)

    # Dedicated hats / very-high transients (separate from HIGH_BAND)
    ENABLE_HATS = True
    HAT_BAND = (8000.0, 16000.0)
    HAT_GAIN = 2.2
    HAT_BAND_Z = 1.4
    HAT_STRENGTH_Z0 = 0.6
    HAT_STRENGTH_ZRANGE = 2.2
    HAT_ATTACK_RATE = 10.0
    HAT_RELEASE_RATE = 3.0
    HAT_LP_RC = 0.025
    HAT_SEL_FRAC = 0.35
    HAT_RESELECT_EVERY_BEATS = 8
    HAT_WHITE_TICK = 0.12          # subtle whiteness on hats (0 disables). Keep low to avoid strobing.

    # Kick detector sensitivity (tuning knobs)
    KICK_MIN_RMS = 0.0008        # was 0.0020; lower helps quiet system-audio captures
    KICK_COOLDOWN_S = 0.09       # minimum time between kicks
    # z-score thresholds (lower = more sensitive)
    KICK_ONSET_BASS_DZ = 2.1
    KICK_ONSET_BASS_DBZ = 0.5
    KICK_RATIO_BASS_Z = 1.3
    KICK_RATIO_FLUX_Z = 0.7
    KICK_ABS_BASSRAW_Z = 1.8
    KICK_ABS_FLUX_Z = 0.4
    KICK_BAND_Z = 1.7            # primary detector in kick band (lower = more sensitive)
    KICK_BAND_SUPPORT_Z = 0.2    # require a tiny bit of flux or bass_db support to avoid pure sub wobble false positives
    # Strength mapping (for visuals)
    KICK_STRENGTH_Z0 = 0.9       # z where strength starts rising (was ~1.2)
    KICK_STRENGTH_ZRANGE = 2.8   # z range to reach full strength (was ~3.2)
    # Activity scaling for strengths (avoid muting everything on low-level inputs)
    ACTIVITY_RMS0 = 0.0012
    ACTIVITY_RMS_RANGE = 0.010
    ACTIVITY_MIN_SCALE = 0.35    # never scale strengths below this (prevents "no response" feeling)

    # Palette shaping (make per-light colors more obviously distinct)
    PALETTE_HUE_SPREAD = 1.8     # >1 spreads clustered hues away from their mean; 1.0 leaves palettes as-is
    PALETTE_MIN_HUE_SEP = 0.08   # minimum circular hue separation between palette colors (0..0.5)
    PALETTE_WRAP_OFFSET = 0.05   # extra hue shift when group size > palette size


class FirstOrderFilter:
    def __init__(self, x0, rc, dt, initialized=True):
        self.x = x0
        self.dt = dt
        self.update_alpha(rc)
        self.initialized = initialized

    def update_alpha(self, rc):
        self.alpha = self.dt / (rc + self.dt)

    def update(self, x):
        if self.initialized:
            self.x = (1.0 - self.alpha) * self.x + self.alpha * x
        else:
            self.initialized = True
            self.x = x
        return self.x

# Optional debug preview UI
try:
    import pygame  # type: ignore
except Exception:
    pygame = None  # type: ignore

# DTLS (Hue Entertainment uses DTLS 1.2 + PSK cipher)
# Hue supports DTLS 1.2 with "TLS_PSK_WITH_AES_128_GCM_SHA256".
try:
    from mbedtls.tls import DTLSConfiguration, ClientContext  # type: ignore
    from mbedtls.tls import DTLSVersion  # type: ignore
    from mbedtls.tls import ciphers_available  # type: ignore
    from mbedtls.tls import TLSWrappedSocket  # type: ignore
    from mbedtls._tls import HandshakeStep, WantReadError, WantWriteError  # type: ignore
except Exception as e:
    raise SystemExit(
        "\nERROR: Failed to import python-mbedtls DTLS classes.\n"
        "Install with: pip install python-mbedtls\n"
        "If it still fails, your venv may have a conflicting 'mbedtls' package.\n"
        f"Import error: {e}\n"
    )


# -----------------------------
# Utilities
# -----------------------------

DEFAULT_CREDS_FILE = os.path.join(os.path.dirname(__file__), "hue_creds.json")

_HEX_RE = re.compile(r"^[0-9a-fA-F]+$")


def load_creds(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_creds(path: str, data: dict) -> None:
    # Best-effort: keep perms tighter than default.
    # NOTE: This is still plaintext; treat it like a password.
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    try:
        os.chmod(tmp, 0o600)
    except Exception:
        pass
    os.replace(tmp, path)


def update_creds_file(path: str, updates: dict) -> None:
    """
    Load+update+save creds file if it exists; otherwise create it.
    """
    base = {}
    try:
        if path and os.path.exists(path):
            base = load_creds(path)
    except Exception:
        base = {}
    base.update(updates)
    save_creds(path, base)


def _clientkey_to_hex_and_b64(clientkey: str) -> Tuple[str, str]:
    """
    Hue v1 registration (`generateclientkey=true`) returns `clientkey` as base64
    on many bridges, but some firmware returns a hex string.
    Normalize to:
      - clientkey_hex: lowercase hex bytes (PSK)
      - clientkey_b64: standard base64 encoding of those bytes
    """
    ck = (clientkey or "").strip()
    if not ck:
        raise ValueError("empty clientkey")

    # If it looks like hex, treat as hex bytes.
    if _HEX_RE.match(ck) and (len(ck) % 2 == 0) and len(ck) >= 16:
        raw = bytes.fromhex(ck)
        return raw.hex(), base64.b64encode(raw).decode("ascii")

    # Otherwise treat as base64.
    raw = base64.b64decode(ck)
    return raw.hex(), base64.b64encode(raw).decode("ascii")


def maybe_migrate_creds(creds: dict) -> Tuple[dict, bool]:
    """
    Best-effort migration for older/buggy creds files.
    Returns (new_creds, changed).
    """
    changed = False
    c = dict(creds or {})

    # Some buggy files stored clientkey in hex under clientkey_b64; fix it.
    ck_b64 = c.get("clientkey_b64")
    ck_hex = c.get("clientkey_hex")
    if isinstance(ck_b64, str) and (_HEX_RE.match(ck_b64.strip() or "") is not None):
        s = ck_b64.strip()
        if (len(s) % 2 == 0) and len(s) >= 16:
            # If clientkey_hex is missing or doesn't match the obvious hex candidate, migrate.
            if not isinstance(ck_hex, str) or ck_hex.strip().lower() != s.lower():
                raw = bytes.fromhex(s)
                c["clientkey_hex"] = raw.hex()
                c["clientkey_b64"] = base64.b64encode(raw).decode("ascii")
                changed = True

    return c, changed


def hue_v1_register_with_button(bridge_ip: str, timeout_s: float, devicetype: str = "hue_sync") -> dict:
    """
    Registers with the Hue bridge v1 endpoint.

    - Press the physical link button on the bridge first.
    - Returns:
      - username: v1 username (often also valid as the CLIP v2 application key)
      - clientkey_b64: base64 clientkey (as returned by the bridge)
      - clientkey_hex: hex-decoded PSK bytes
    """
    url = f"http://{bridge_ip}/api"
    payload = {"devicetype": devicetype, "generateclientkey": True}
    deadline = time.time() + timeout_s
    last_err = None

    while time.time() < deadline:
        try:
            r = requests.post(url, json=payload, timeout=5.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            last_err = e
            time.sleep(1.0)
            continue

        # Expected response: [{"success": {"username": "...", "clientkey": "..."}}]
        # or [{"error": {...}}] with type 101 ("link button not pressed").
        if isinstance(data, list) and data:
            item = data[0]
            if "success" in item:
                success = item["success"] or {}
                username = success.get("username")
                clientkey_raw = success.get("clientkey")
                if not username or not clientkey_raw:
                    raise RuntimeError(f"Unexpected registration success payload: {success!r}")
                try:
                    clientkey_hex, clientkey_b64 = _clientkey_to_hex_and_b64(str(clientkey_raw))
                except Exception as e:
                    raise RuntimeError(f"Failed to normalize clientkey -> hex: {e}")
                return {
                    "username": username,
                    "clientkey_b64": clientkey_b64,
                    "clientkey_hex": clientkey_hex,
                }
            if "error" in item:
                last_err = item["error"]

        time.sleep(1.0)

    raise SystemExit(
        f"ERROR: Timed out waiting for bridge link button ({timeout_s:.0f}s). "
        f"Last response/error: {last_err!r}"
    )


def clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def hsv_to_rgb(h: float, s: float, v: float) -> Tuple[float, float, float]:
    # h in [0,1), s,v in [0,1]
    h = (h % 1.0 + 1.0) % 1.0
    s = clamp(s, 0.0, 1.0)
    v = clamp(v, 0.0, 1.0)
    i = int(h * 6.0)
    f = h * 6.0 - i
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    i = i % 6
    if i == 0: return v, t, p
    if i == 1: return q, v, p
    if i == 2: return p, v, t
    if i == 3: return p, q, v
    if i == 4: return t, p, v
    return v, p, q


def srgb_gamma(x: float) -> float:
    # mild gamma to make lows more visible
    x = clamp(x, 0.0, 1.0)
    return x ** 2.2


def rgb_to_u16(r: float, g: float, b: float, gamma: bool = True) -> Tuple[int, int, int]:
    if gamma:
        r, g, b = srgb_gamma(r), srgb_gamma(g), srgb_gamma(b)
    return (int(clamp(r, 0, 1) * 65535),
            int(clamp(g, 0, 1) * 65535),
            int(clamp(b, 0, 1) * 65535))


# -----------------------------
# Hue Entertainment Streamer
# -----------------------------

class _PatchedTLSWrappedSocket(TLSWrappedSocket):
    """
    Mirrors hue_entertainment_pykit handshake logic.

    python-mbedtls DTLS can require explicit send/recv driving with retransmits
    for ClientHello; hue_entertainment_pykit patches TLSWrappedSocket.do_handshake
    to implement this.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._handshake_retries = 0

    def do_handshake(self, *args):
        if args and self.type is not socket.SOCK_DGRAM:
            raise OSError(107, "Transport endpoint is not connected")

        if len(args) == 0:
            flags, address = 0, None
        elif len(args) == 1:
            flags, address = 0, args[0]
        elif len(args) == 2:
            assert isinstance(args[0], int)
            flags, address = args
        else:
            raise TypeError("do_handshake() takes 0, 1, or 2 arguments")

        while self._handshake_state is not HandshakeStep.HANDSHAKE_OVER:
            try:
                self._buffer.do_handshake()
            except WantReadError as exc:
                if address is None:
                    data = self._socket.recv(TLSWrappedSocket.CHUNK_SIZE, flags)
                else:
                    data, addr = self._socket.recvfrom(TLSWrappedSocket.CHUNK_SIZE, flags)
                    if addr != address:
                        raise OSError(107, "Transport endpoint is not connected") from exc
                self._buffer.receive_from_network(data)
            except WantWriteError as exc:
                in_transit = self._buffer.peek_outgoing(TLSWrappedSocket.CHUNK_SIZE)
                if address is None:
                    amt = self._socket.send(in_transit, flags)
                else:
                    amt = self._socket.sendto(in_transit, flags, address)
                self._buffer.consume_outgoing(amt)

                self._handshake_retries += 1
                # Resend ClientHello a couple times to handle lossy DTLS starts.
                if self._handshake_retries < 3:
                    time.sleep(0.3)
                    if address is None:
                        amt = self._socket.send(in_transit, flags)
                    else:
                        amt = self._socket.sendto(in_transit, flags, address)
                    self._buffer.consume_outgoing(amt)
                if self._handshake_retries > 3:
                    raise RuntimeError("DTLS handshake retries exceeded") from exc


class HueEntertainmentStreamer:
    """
    Streams Hue Entertainment frames over DTLS/UDP to bridge:2100
    Packet format (v2):
      "HueStream" (8 bytes)
      header_rest (8 bytes)
      entertainment_id UUID string (36 bytes ASCII)
      channels: up to 20 * (1 + 2 + 2 + 2) bytes => channel_id + RGB16
    """

    def __init__(
        self,
        bridge_ip: str,
        clip_app_key: str,
        psk_identity: str,
        clientkey_hex: str,
        entertainment_id: str,
        channel_ids: List[int],
        dtls_timeout_s: float = 10.0,
        dtls_backend: str = "pykit",
    ) -> None:
        self.bridge_ip = bridge_ip
        # `clip_app_key` is used for HTTPS CLIP v2 requests (hue-application-key header).
        # `psk_identity` is used as the DTLS PSK identity for the Entertainment stream.
        #
        # In some Hue setups these are the same string; in others, the DTLS PSK identity
        # must be the legacy v1 "username" that was created with `generateclientkey=true`.
        self.clip_app_key = clip_app_key
        self.psk_identity = psk_identity
        self.clientkey_hex = clientkey_hex.strip().lower()
        self.entertainment_id = entertainment_id.strip()
        self.channel_ids = channel_ids[:]  # list of channel numbers (0..)
        self.dtls_timeout_s = dtls_timeout_s
        self.dtls_backend = dtls_backend

        if len(self.entertainment_id) != 36:
            raise ValueError("--ent-id must be a 36-char UUID string (with hyphens), e.g. 6eaf...f290")

        if len(self.channel_ids) < 1 or len(self.channel_ids) > 20:
            raise ValueError("channel_ids must be 1..20 for Hue Entertainment streaming")

        # Hue expects PSK bytes from clientkey hex.
        try:
            self.psk = bytes.fromhex(self.clientkey_hex)
        except ValueError:
            raise ValueError("--clientkey-hex must be hex (no 0x), e.g. 'a1b2...'. If you have base64, convert it.")

        # Build DTLS socket
        self._sock = None
        self._seq = 0
        self._pykit_dtls = None

    def connect(self) -> None:
        if self.dtls_backend == "pykit":
            self._connect_pykit()
            return

        if self.dtls_backend != "mbedtls":
            raise RuntimeError(f"Unknown --dtls-backend {self.dtls_backend!r} (expected 'pykit' or 'mbedtls')")

        # Hue DTLS requirements: DTLS 1.2, PSK, AES_128_GCM_SHA256.
        # python-mbedtls expects `ciphers` as a sequence of cipher-suite names
        # (e.g. ["TLS-PSK-WITH-AES-128-GCM-SHA256"]). Passing a single string is
        # treated as an iterable of characters and fails with "unsupported ciphers".
        available = list(ciphers_available())
        avail_set = set(available)
        # Match hue_entertainment_pykit defaults first (AES-256-GCM-SHA384).
        preferred = [
            "TLS-PSK-WITH-AES-256-GCM-SHA384",
            "TLS-DHE-PSK-WITH-AES-256-GCM-SHA384",
            # Fallbacks (still PSK)
            "TLS-PSK-WITH-AES-128-GCM-SHA256",
            "TLS-DHE-PSK-WITH-AES-128-GCM-SHA256",
            "TLS-PSK-WITH-AES-128-CCM",
            "TLS-DHE-PSK-WITH-AES-128-CCM",
        ]
        # Broader fallback: offer any available PSK cipher with AES-128/256 (GCM/CCM/CBC),
        # keeping preferred at the front to guide selection.
        broad = [
            c
            for c in available
            if ("PSK" in c)
            and (
                ("AES-128" in c)
                or ("AES_128" in c)
                or ("AES-256" in c)
                or ("AES_256" in c)
            )
        ]
        ciphers = [c for c in preferred if c in avail_set] + [c for c in broad if c not in preferred]
        # stash for debugging output
        self._offered_ciphers = ciphers
        if not ciphers:
            raise RuntimeError(
                "No supported PSK DTLS cipher suites found in python-mbedtls build. "
                "Expected at least 'TLS-PSK-WITH-AES-128-GCM-SHA256'."
            )

        conf = DTLSConfiguration(
            pre_shared_key=(self.psk_identity, self.psk),
            ciphers=ciphers,
            validate_certificates=False,
            lowest_supported_version=DTLSVersion.DTLSv1_2,
            highest_supported_version=DTLSVersion.DTLSv1_2,
        )
        ctx = ClientContext(conf)
        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.settimeout(self.dtls_timeout_s)
        udp.connect((self.bridge_ip, 2100))
        buf = ctx.wrap_buffers(server_hostname=self.bridge_ip)
        sock = _PatchedTLSWrappedSocket(udp, buf)
        sock.do_handshake()
        self._sock = sock

    def _connect_pykit(self) -> None:
        """
        Use hue_entertainment_pykit DTLS implementation (known-good in this environment).
        This avoids module-name collisions by importing hue_entertainment_pykit first (it patches sys.path).
        """
        # Import the module first so its sys.path/sys.modules conflict workaround takes effect.
        import hue_entertainment_pykit  # noqa: F401

        from models.bridge import Bridge  # provided by the pykit distribution
        from network.dtls import Dtls  # provided by the pykit distribution

        # pykit DTLS uses Bridge.hue_app_id as PSK identity and Bridge.client_key as the clientkey string.
        # We pass the clientkey hex (normalized by our CLI/creds).
        bridge = Bridge(ip_address=self.bridge_ip, hue_app_id=self.psk_identity, client_key=self.clientkey_hex)
        dtls = Dtls(bridge)
        # best-effort timeout alignment
        try:
            dtls._sock_timeout = int(max(1, round(self.dtls_timeout_s)))  # type: ignore[attr-defined]
        except Exception:
            pass

        if getattr(self, "_offered_ciphers", None) is not None:
            # reuse existing debug infra
            pass
        dtls.do_handshake()
        sock = dtls.get_socket()
        if sock is None:
            raise RuntimeError("pykit DTLS socket is None after handshake")
        self._pykit_dtls = dtls
        self._sock = sock

    def close(self) -> None:
        try:
            if self._sock:
                self._sock.close()
        finally:
            self._sock = None
            self._pykit_dtls = None

    def _build_packet(self, rgb16_by_channel: Dict[int, Tuple[int, int, int]]) -> bytes:
        # Match hue_entertainment_pykit wire format:
        #   "HueStream" (8)
        #   version (2): 0x02 0x00
        #   sequence (1)
        #   reserved (2): 0x00 0x00
        #   color_space (1): 0x00 for rgb, 0x01 for xyb
        #   reserved2 (1): 0x00
        protocol = b"HueStream"  # 8 bytes
        self._seq = (self._seq + 1) & 0xFF
        version = bytes([0x02, 0x00])
        sequence = bytes([self._seq])
        reserved = b"\x00\x00"
        color_space = b"\x00"  # rgb
        reserved2 = b"\x00"
        ent_id = self.entertainment_id.encode("utf-8")  # 36 bytes ASCII UUID

        # Channel payloads: for each channel id, pack B + R16 + G16 + B16 in big endian
        out = bytearray()
        out += protocol
        out += version
        out += sequence
        out += reserved
        out += color_space
        out += reserved2
        out += ent_id

        for ch in self.channel_ids:
            r, g, b = rgb16_by_channel.get(ch, (0, 0, 0))
            out += struct.pack(">BHHH", int(ch) & 0xFF, int(r) & 0xFFFF, int(g) & 0xFFFF, int(b) & 0xFFFF)

        return bytes(out)

    def send_frame(self, rgb16_by_channel: Dict[int, Tuple[int, int, int]]) -> None:
        if not self._sock:
            raise RuntimeError("DTLS socket not connected")
        pkt = self._build_packet(rgb16_by_channel)
        self._sock.send(pkt)


# -----------------------------
# Minimal CLIP v2 start/stop
# -----------------------------

def clip_v2_start_stop_entertainment(bridge_ip: str, app_key: str, ent_id: str, action: str) -> None:
    # PUT https://<HUB>/clip/v2/resource/entertainment_configuration/<id> {"action":"start"}
    url = f"https://{bridge_ip}/clip/v2/resource/entertainment_configuration/{ent_id}"
    headers = {"hue-application-key": app_key, "Content-Type": "application/json"}
    r = requests.put(url, headers=headers, json={"action": action}, timeout=5.0, verify=False)
    # If this fails, streaming might still work if you already started via app,
    # but usually the area will stop quickly if not started.
    if r.status_code >= 300:
        raise RuntimeError(f"Failed to {action} entertainment area: HTTP {r.status_code}: {r.text}")


def clip_v2_get_entertainment_configuration(bridge_ip: str, app_key: str, ent_id: str) -> dict:
    url = f"https://{bridge_ip}/clip/v2/resource/entertainment_configuration/{ent_id}"
    headers = {"hue-application-key": app_key}
    r = requests.get(url, headers=headers, timeout=5.0, verify=False)
    if r.status_code >= 300:
        raise RuntimeError(f"Failed to read entertainment area: HTTP {r.status_code}: {r.text}")
    return r.json()


def clip_v2_get_hue_application_id(bridge_ip: str, app_key: str) -> str:
    """
    Hue returns a `hue-application-id` response header from `GET /auth/v1`.
    hue_entertainment_pykit uses this value as the DTLS PSK identity.
    """
    url = f"https://{bridge_ip}/auth/v1"
    headers = {"hue-application-key": app_key}
    r = requests.get(url, headers=headers, timeout=5.0, verify=False)
    if r.status_code >= 300:
        raise RuntimeError(f"Failed to fetch hue-application-id: HTTP {r.status_code}: {r.text}")
    hue_app_id = r.headers.get("hue-application-id")
    if not hue_app_id:
        raise RuntimeError("Bridge did not return `hue-application-id` header on /auth/v1")
    return hue_app_id


# -----------------------------
# Audio Analysis (fast, real-time)
# -----------------------------

@dataclass
class Features:
    rms: float
    bass: float
    mid: float
    treble: float
    centroid: float
    flux: float
    flatness: float
    kick: bool
    riser: float  # 0..1
    onset: float  # onset strength proxy (for BPM tracking)
    # `*_strength` are VISUAL strengths (may include minimum scaling so the show doesn't feel dead).
    kick_strength: float = 0.0  # 0..1 intensity from kick low-frequency band (used for flashing)
    high_strength: float = 0.0  # 0..1 combined high band (kick click + hats)
    hat_strength: float = 0.0   # 0..1 very-high transient strength (hats/air band)
    # Raw strengths for BPM/beat tracking (NO minimum scaling; safe to gate by activity)
    kick_strength_raw: float = 0.0
    high_strength_raw: float = 0.0
    hat_strength_raw: float = 0.0
    activity: float = 0.0       # 0..1 loudness/activity gate (for beat/BPM + noise suppression)
    beat: bool = False
    beat_phase: float = 0.0  # 0..1 position within beat
    beat_idx: int = 0
    bpm_est: float = 0.0
    bpm_conf: float = 0.0
    double_burst: bool = False  # short "double-time" burst for moving spotlight accents
    onset_rate: float = 0.0  # onset events per second (beat/drive proxy)


class FastAudioAnalyzer:
    """
    Lightweight analyzer designed for 50Hz control loops:
    - RMS energy
    - 3-band energy (bass/mid/treble) from FFT magnitude
    - Spectral centroid
    - Spectral flux (onset-ish)
    - Spectral flatness (tonal vs noisy)
    - Simple kick detector (adaptive bass + flux)
    - Simple riser detector (centroid ramp)
    """

    def __init__(
        self,
        sr: int,
        n_fft: int,
        bands: Tuple[Tuple[float, float], ...],
        perceptual_weighting: str = "none",
    ) -> None:
        self.sr = sr
        self.n_fft = n_fft
        self.window = np.hanning(n_fft).astype(np.float32)
        self.prev_mag: Optional[np.ndarray] = None

        # band edges in Hz
        self.bands = bands
        self.freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sr).astype(np.float32)
        self.band_masks = [((self.freqs >= lo) & (self.freqs < hi)) for (lo, hi) in self.bands]

        # Perceptual weighting (helps match what humans hear; reduces sub-bass over-driving visuals)
        self.perceptual_weighting = str(perceptual_weighting or "none").lower()
        self.mag_weights = np.ones_like(self.freqs, dtype=np.float32)
        if self.perceptual_weighting in ("a", "a-weighting", "a_weighting"):
            f = self.freqs.astype(np.float64)
            f2 = f * f
            # IEC/CD 1672 A-weighting (dB)
            ra_num = (12200.0**2) * (f2 * f2)
            ra_den = (f2 + 20.6**2) * (f2 + 12200.0**2) * np.sqrt((f2 + 107.7**2) * (f2 + 737.9**2))
            ra = ra_num / (ra_den + 1e-30)
            a_db = 2.0 + 20.0 * np.log10(ra + 1e-30)
            # Convert dB to linear amplitude scaling for magnitude spectrum
            w = np.power(10.0, a_db / 20.0)
            w[~np.isfinite(w)] = 0.0
            w = np.clip(w, 0.0, 16.0)  # keep numerics sane; we only need relative weighting
            self.mag_weights = w.astype(np.float32)

        # EMA state
        self.ema_rms = 0.0
        self.ema_bass = 0.0
        self.ema_flux = 0.0
        self.ema_centroid = 0.0
        self.ema_flatness = 0.0

        # extra kick robustness (absolute bass energy)
        self.ema_bass_raw = 0.0
        self.bass_raw_var = 1e-6
        self.ema_bass_db = 0.0
        self.bass_db_var = 1e-6
        self.prev_bass_db: Optional[float] = None
        self.ema_db_diff = 0.0
        self.db_diff_var = 1e-6
        # Kick-band (low) for visual flashing (EDM kick often spans ~40..250Hz incl. fundamental + punch)
        klo, khi = ShowOptions.KICK_BAND
        self.kick_lo_mask = (self.freqs >= float(klo)) & (self.freqs < float(khi))
        self.ema_kick_lo = 0.0
        self.kick_lo_var = 1e-6
        # Combined "high" band (kick click + hats) for other lights
        hlo, hhi = ShowOptions.HIGH_BAND
        self.high_lo_hz = float(hlo)
        self.high_hi_hz = float(hhi)
        self.high_mask = (self.freqs >= self.high_lo_hz) & (self.freqs < self.high_hi_hz)
        self.ema_high = 0.0
        self.high_var = 1e-6
        # Very-high band (hats/air) for sparkle group
        hlo2, hhi2 = getattr(ShowOptions, "HAT_BAND", (8000.0, 16000.0))
        self.hat_lo_hz = float(hlo2)
        self.hat_hi_hz = float(hhi2)
        self.hat_mask = (self.freqs >= self.hat_lo_hz) & (self.freqs < self.hat_hi_hz)
        self.ema_hat = 0.0
        self.hat_var = 1e-6
        # Debug spectrum snapshot (for pygame spectrum window)
        self.last_mag_raw: Optional[np.ndarray] = None
        # Debug scalars (for pygame overlay / tuning)
        self.dbg_bass_db = 0.0
        self.dbg_bass_db_z = 0.0
        self.dbg_db_diff = 0.0
        self.dbg_bass_dz = 0.0
        self.dbg_flux = 0.0
        self.dbg_flux_dev = 0.0
        self.dbg_flux_z = 0.0

    def set_kick_band(self, low_hz: float, high_hz: float) -> None:
        lo = float(low_hz)
        hi = float(high_hz)
        lo = max(10.0, min(lo, self.sr / 2 - 1.0))
        hi = max(lo + 1.0, min(hi, self.sr / 2))
        self.kick_lo_mask = (self.freqs >= lo) & (self.freqs < hi)

    def set_high_band(self, low_hz: float, high_hz: float) -> None:
        lo = float(low_hz)
        hi = float(high_hz)
        lo = max(50.0, min(lo, self.sr / 2 - 1.0))
        hi = max(lo + 1.0, min(hi, self.sr / 2))
        self.high_lo_hz = lo
        self.high_hi_hz = hi
        self.high_mask = (self.freqs >= lo) & (self.freqs < hi)

    def set_hat_band(self, low_hz: float, high_hz: float) -> None:
        lo = float(low_hz)
        hi = float(high_hz)
        lo = max(1000.0, min(lo, self.sr / 2 - 1.0))
        hi = max(lo + 1.0, min(hi, self.sr / 2))
        self.hat_lo_hz = lo
        self.hat_hi_hz = hi
        self.hat_mask = (self.freqs >= lo) & (self.freqs < hi)

        # For kick/riser
        self.bass_var = 1e-6
        self.flux_var = 1e-6
        self.centroid_slope = 0.0
        self.last_centroids = []

        self.last_kick_t = 0.0
        self.kick_intervals = []  # seconds

    def analyze(self, mono: np.ndarray, now_s: float) -> Features:
        # mono: float32, length >= n_fft preferred (we'll take last n_fft samples)
        if mono.size < self.n_fft:
            x = np.zeros(self.n_fft, dtype=np.float32)
            x[-mono.size:] = mono.astype(np.float32, copy=False)
        else:
            x = mono[-self.n_fft:].astype(np.float32, copy=False)

        # RMS
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        # FFT mags
        xw = x * self.window
        spec = np.fft.rfft(xw)
        mag_raw = np.abs(spec).astype(np.float32) + 1e-12
        self.last_mag_raw = mag_raw
        # Use perceptually-weighted spectrum for visual features (bands/flux/centroid/flatness)
        mag = mag_raw * self.mag_weights

        # Band energies (normalized-ish)
        band_vals = []
        for mask in self.band_masks:
            v = float(np.mean(mag[mask])) if np.any(mask) else 0.0
            band_vals.append(v)

        bass_raw, mid_raw, treble_raw = band_vals[0], band_vals[1], band_vals[2]
        total = bass_raw + mid_raw + treble_raw + 1e-9
        bass = bass_raw / total
        mid = mid_raw / total
        treble = treble_raw / total

        # Raw bass energy for kick detection (avoid A-weighting suppressing real kicks)
        bass_mask = self.band_masks[0]
        bass_raw_kick = float(np.mean(mag_raw[bass_mask])) if np.any(bass_mask) else 0.0
        kick_lo_raw = float(np.mean(mag_raw[self.kick_lo_mask])) if np.any(self.kick_lo_mask) else 0.0
        # High band is spiky (hats/clicks). Using a plain mean across 5k–15k dilutes transients.
        # Use mean of top ~5% bins to better represent "attack" energy.
        if np.any(self.high_mask):
            hb = mag_raw[self.high_mask]
            if hb.size > 8:
                k = max(1, int(hb.size * 0.05))
                top = np.partition(hb, hb.size - k)[hb.size - k :]
                high_raw = float(np.mean(top))
            else:
                high_raw = float(np.mean(hb))
        else:
            high_raw = 0.0

        # Hat band (very-high transients): use mean of top ~10% bins to capture crisp hats without dilution.
        if np.any(self.hat_mask):
            hb2 = mag_raw[self.hat_mask]
            if hb2.size > 12:
                k2 = max(1, int(hb2.size * 0.10))
                top2 = np.partition(hb2, hb2.size - k2)[hb2.size - k2 :]
                hat_raw = float(np.mean(top2))
            else:
                hat_raw = float(np.mean(hb2))
        else:
            hat_raw = 0.0

        # Spectral centroid
        centroid = float(np.sum(self.freqs * mag) / np.sum(mag))

        # Spectral flux (positive changes)
        if self.prev_mag is None:
            flux = 0.0
        else:
            d = mag - self.prev_mag
            flux = float(np.mean(np.maximum(d, 0.0)) / (np.mean(mag) + 1e-9))
        self.prev_mag = mag

        # Spectral flatness (noisy ~ 1, tonal ~ 0)
        geo = float(np.exp(np.mean(np.log(mag))))
        ari = float(np.mean(mag))
        flatness = float(geo / (ari + 1e-12))

        # EMA smoothing
        def ema(prev: float, x_: float, a: float) -> float:
            return prev * (1.0 - a) + x_ * a

        self.ema_rms = ema(self.ema_rms, rms, 0.15)
        self.ema_bass = ema(self.ema_bass, bass, 0.15)
        self.ema_flux = ema(self.ema_flux, flux, 0.20)
        self.ema_centroid = ema(self.ema_centroid, centroid, 0.10)
        self.ema_flatness = ema(self.ema_flatness, flatness, 0.10)
        self.ema_bass_raw = ema(self.ema_bass_raw, bass_raw_kick, 0.12)
        self.ema_kick_lo = ema(self.ema_kick_lo, kick_lo_raw, 0.10)
        self.ema_high = ema(self.ema_high, high_raw, 0.16)
        self.ema_hat = ema(self.ema_hat, hat_raw, 0.18)

        # Bass energy in log domain (stabilizes across volume levels)
        bass_db = float(np.log(bass_raw_kick + 1e-12))
        self.ema_bass_db = ema(self.ema_bass_db, bass_db, 0.12)
        bass_db_dev = bass_db - self.ema_bass_db
        self.bass_db_var = ema(self.bass_db_var, bass_db_dev * bass_db_dev, 0.06)
        bass_db_z = bass_db_dev / (math.sqrt(self.bass_db_var) + 1e-6)

        # Derivative z-score (onset-like)
        if self.prev_bass_db is None:
            db_diff = 0.0
        else:
            db_diff = bass_db - self.prev_bass_db
        self.prev_bass_db = bass_db
        self.ema_db_diff = ema(self.ema_db_diff, db_diff, 0.20)
        db_diff_dev = db_diff - self.ema_db_diff
        self.db_diff_var = ema(self.db_diff_var, db_diff_dev * db_diff_dev, 0.10)
        bass_dz = db_diff_dev / (math.sqrt(self.db_diff_var) + 1e-6)

        # Adaptive kick detector:
        # Robust kick detector:
        # - bass onset in log domain (volume-invariant-ish)
        # - optional flux support (helps separate kick from steady sub)
        # - cooldown so it doesn't machine-gun
        bass_dev = bass - self.ema_bass
        flux_dev = flux - self.ema_flux

        # Onset proxy:
        # Autocorrelation BPM works best when the onset envelope has strong periodic structure.
        # Pure spectral flux can be too noisy on EDM (hats) and too weak on heavy compression.
        # Mix in the already-stabilized kick/high "strength" signals so beats are more consistent.
        onset = float(0.35 * max(0.0, db_diff) + 0.9 * max(0.0, flux_dev))

        # Track variance (very rough)
        self.bass_var = ema(self.bass_var, bass_dev * bass_dev, 0.05)
        self.flux_var = ema(self.flux_var, flux_dev * flux_dev, 0.05)

        bass_z = bass_dev / (math.sqrt(self.bass_var) + 1e-6)
        flux_z = flux_dev / (math.sqrt(self.flux_var) + 1e-6)

        # Save debug values (so the pygame overlay can show *why* onset/kick are firing)
        self.dbg_bass_db = float(bass_db)
        self.dbg_bass_db_z = float(bass_db_z)
        self.dbg_db_diff = float(db_diff)
        self.dbg_bass_dz = float(bass_dz)
        self.dbg_flux = float(flux)
        self.dbg_flux_dev = float(flux_dev)
        self.dbg_flux_z = float(flux_z)

        # Bass absolute energy spike (helps when bass ratio stays constant but amplitude jumps)
        bass_raw_dev = bass_raw_kick - self.ema_bass_raw
        self.bass_raw_var = ema(self.bass_raw_var, bass_raw_dev * bass_raw_dev, 0.05)
        bass_raw_z = bass_raw_dev / (math.sqrt(self.bass_raw_var) + 1e-6)

        # Kick-band strength (for visuals): how much the kick band pops above its EMA
        kick_lo_dev = kick_lo_raw - self.ema_kick_lo
        self.kick_lo_var = ema(self.kick_lo_var, kick_lo_dev * kick_lo_dev, 0.05)
        kick_lo_z = kick_lo_dev / (math.sqrt(self.kick_lo_var) + 1e-6)
        # Map to 0..1; tuned to be punchy but not always-on (level-above-EMA, not raw volume)
        z0 = float(getattr(ShowOptions, "KICK_STRENGTH_Z0", 0.9))
        zrange = float(getattr(ShowOptions, "KICK_STRENGTH_ZRANGE", 2.8))
        kick_strength = float(clamp((kick_lo_z - z0) / max(1e-6, zrange), 0.0, 1.0))

        # High-band strength (for other lights): pop above EMA in the combined high band
        high_dev = high_raw - self.ema_high
        self.high_var = ema(self.high_var, high_dev * high_dev, 0.08)
        high_z = high_dev / (math.sqrt(self.high_var) + 1e-6)
        # Slightly more sensitive than low kick; high band often has smaller absolute energy.
        high_strength = float(clamp((high_z - 0.6) / 2.2, 0.0, 1.0))

        # Hat-band strength (for hat sparkle group)
        hat_dev = hat_raw - self.ema_hat
        self.hat_var = ema(self.hat_var, hat_dev * hat_dev, 0.10)
        hat_z = hat_dev / (math.sqrt(self.hat_var) + 1e-6)
        hz0 = float(getattr(ShowOptions, "HAT_STRENGTH_Z0", 0.6))
        hzr = float(getattr(ShowOptions, "HAT_STRENGTH_ZRANGE", 2.2))
        hat_strength = float(clamp((hat_z - hz0) / max(1e-6, hzr), 0.0, 1.0))

        # Activity gate:
        # In very quiet sections, the EMA/variance can get tiny, making z-scores jittery and causing false positives.
        # We keep a minimum scaling for *visual* strengths so it doesn't feel dead,
        # but we will still suppress onset (used for beat/BPM) by the raw activity value.
        rms0 = float(getattr(ShowOptions, "ACTIVITY_RMS0", 0.0012))
        rmsr = float(getattr(ShowOptions, "ACTIVITY_RMS_RANGE", 0.010))
        activity = float(clamp((self.ema_rms - rms0) / max(1e-6, rmsr), 0.0, 1.0))
        min_scale = float(getattr(ShowOptions, "ACTIVITY_MIN_SCALE", 0.35))
        scale = float(min_scale + (1.0 - min_scale) * activity)
        # Keep raw strengths for beat/BPM.
        kick_strength_raw = float(kick_strength)
        high_strength_raw = float(high_strength)
        hat_strength_raw = float(hat_strength)
        # Visual strengths include minimum scaling.
        kick_strength_vis = float(kick_strength_raw) * float(scale)
        high_strength_vis = float(high_strength_raw) * float(scale)
        hat_strength_vis = float(hat_strength_raw) * float(scale)

        # Final onset envelope for BPM/beat:
        # - add rhythmic components (kick/high/hat) so BPM doesn't depend on flux alone
        # - suppress it in quiet sections by the raw activity (so we don't hallucinate beats)
        onset = float(onset + 0.70 * kick_strength_raw + 0.25 * high_strength_raw + 0.15 * hat_strength_raw)
        onset = float(onset * activity)

        kick = False
        cooldown = float(getattr(ShowOptions, "KICK_COOLDOWN_S", 0.09))
        if (now_s - self.last_kick_t) > cooldown:
            # Minimum activity gate (very low to work with quiet/system audio)
            if self.ema_rms > float(getattr(ShowOptions, "KICK_MIN_RMS", 0.0008)):
                # Primary: kick-band pop above baseline (works well in EDM even with steady basslines)
                kick_band_z = float(getattr(ShowOptions, "KICK_BAND_Z", 1.7))
                sup_z = float(getattr(ShowOptions, "KICK_BAND_SUPPORT_Z", 0.2))
                kickband_ok = (kick_lo_z > kick_band_z) and ((flux_z > sup_z) or (bass_db_z > sup_z))

                # Secondary: strong bass onset (derivative) + bass energy above baseline
                onset_ok = (bass_dz > float(getattr(ShowOptions, "KICK_ONSET_BASS_DZ", 2.1))) and (bass_db_z > float(getattr(ShowOptions, "KICK_ONSET_BASS_DBZ", 0.5)))
                # Tertiary: ratio spike + some flux
                ratio_ok = (bass_z > float(getattr(ShowOptions, "KICK_RATIO_BASS_Z", 1.3))) and (flux_z > float(getattr(ShowOptions, "KICK_RATIO_FLUX_Z", 0.7)))
                # Also: absolute bass spike + a bit of flux
                abs_ok = (bass_raw_z > float(getattr(ShowOptions, "KICK_ABS_BASSRAW_Z", 1.8))) and (flux_z > float(getattr(ShowOptions, "KICK_ABS_FLUX_Z", 0.4)))

                if kickband_ok or onset_ok or ratio_ok or abs_ok:
                    kick = True
                    if self.last_kick_t > 0:
                        dt = now_s - self.last_kick_t
                        if 0.18 < dt < 1.2:
                            self.kick_intervals.append(dt)
                            if len(self.kick_intervals) > 12:
                                self.kick_intervals.pop(0)
                    self.last_kick_t = now_s

        # Riser detector: centroid ramp upward + energy present
        self.last_centroids.append(self.ema_centroid)
        if len(self.last_centroids) > 25:  # ~0.5s at 50Hz
            self.last_centroids.pop(0)
        if len(self.last_centroids) >= 8:
            # slope = last - first over window
            slope = (self.last_centroids[-1] - self.last_centroids[0]) / max(len(self.last_centroids) - 1, 1)
            # normalize slope into [0,1] (tuned)
            riser = float(clamp((slope - 5.0) / 60.0, 0.0, 1.0))
        else:
            riser = 0.0

        return Features(
            rms=self.ema_rms,
            bass=bass,
            mid=mid,
            treble=treble,
            centroid=self.ema_centroid,
            flux=self.ema_flux,
            flatness=self.ema_flatness,
            kick=kick,
            kick_strength=kick_strength_vis,
            high_strength=high_strength_vis,
            hat_strength=hat_strength_vis,
            kick_strength_raw=kick_strength_raw,
            high_strength_raw=high_strength_raw,
            hat_strength_raw=hat_strength_raw,
            activity=float(activity),
            riser=riser,
            onset=onset,
        )


# -----------------------------
# Show Engine (EDM-ish)
# -----------------------------

class EpicShow:
    """
    Converts Features -> per-channel RGB16 frames.

    Design goals:
    - Kicks: strobe/flash and palette jump
    - Drops: stronger saturation + wider palette spread
    - Melodic / quiet: smooth movement driven by centroid + tonal detection
    - Risers: hue spin + brightness ramp, then drop flash
    """

    def __init__(self, channel_ids: List[int]) -> None:
        self.channel_ids = channel_ids
        self.n = len(channel_ids)

        self.base_h = 0.10  # starting hue
        self.h_spin = 0.0

        self.strobe = 0.0  # decays
        self.drop_flash = 0.0
        self.last_kick_time = 0.0
        self._prev_kick_time = 0.0
        self._kick_intervals: List[float] = []
        self.kick_bpm = 0.0
        self._double_time_until = 0.0
        self._double_time_hits = 0

        self.energy_ema = 0.0
        self.mode = "chill"  # "chill" or "drop"
        self.mode_hold = 0.0

        # Per-channel motion accents
        self.chase_phase = 0.0
        self.chase_idx = 0

        # Beat-locked accents (driven by bpm_est beat grid)
        self.beat_flash = 0.0
        self.last_beat_time = 0.0
        self.beat_idx = 0
        self.spot_until = 0.0
        self.spot_idx = 0
        self._rng = random.Random(1337)
        self._ch_gain: Dict[int, float] = {ch: (0.88 + 0.24 * self._rng.random()) for ch in self.channel_ids}

        # Sections disabled for now: we run a single beat-synced mode.
        self.section = "beat"

        # Per-role groups (persistent random assignment; reshuffled occasionally)
        self._group_role: Dict[int, str] = {}
        self._last_group_beat_idx = -1
        # Keep this 0 by default so it doesn't feel random; you can re-enable later.
        self._group_reshuffle_every = 0  # beats (0 = never)
        self._reshuffle_groups(force=True)

        # Beat-synced envelopes
        self._beat_pulse = 0.0
        self._hat_pulse = 0.0
        self._mel_pulse = 0.0
        # Single "high kick" envelope for non-kick lights (combines click + hats)
        self._high_pulse = 0.0

        # First-order filters (use requested FirstOrderFilter class)
        # Slight smoothing, gives "momentum" without feeling laggy
        self._v_lp_rc = 0.03  # seconds
        self._v_filt = FirstOrderFilter(0.0, rc=self._v_lp_rc, dt=(1.0 / 50.0), initialized=False)
        self._high_filt = FirstOrderFilter(0.0, rc=self._v_lp_rc, dt=(1.0 / 50.0), initialized=False)

        # Beat-triggered brightness animation (tempo-scaled exponential decay).
        self._beat_anim = 0.0
        self._beat_anim_decay_s = 0.22
        self._last_beat_trigger_t = 0.0
        self._tempo_period_s = 0.50  # updated from bpm_est when confident

        # High-band envelope shaping (separate from main brightness so hats/clicks feel smoother)
        self._high_attack_rate = float(getattr(ShowOptions, "HIGH_ATTACK_RATE", 4.0))
        self._high_release_rate = float(getattr(ShowOptions, "HIGH_RELEASE_RATE", 1.2))
        self._high_lp_rc = float(getattr(ShowOptions, "HIGH_LP_RC", 0.07))

        # Hat-band envelope + filter (separate, very snappy)
        self._hat_pulse = 0.0
        self._hat_attack_rate = float(getattr(ShowOptions, "HAT_ATTACK_RATE", 10.0))
        self._hat_release_rate = float(getattr(ShowOptions, "HAT_RELEASE_RATE", 3.0))
        self._hat_lp_rc = float(getattr(ShowOptions, "HAT_LP_RC", 0.025))
        self._hat_filt = FirstOrderFilter(0.0, rc=self._hat_lp_rc, dt=(1.0 / 50.0), initialized=False)

        # "Show" moment state (buildup / drop / sparkles)
        self._buildup_until = 0.0
        self._drop_env = 0.0
        self._last_riser = 0.0
        # Per-channel white sparkle envelope (0..1)
        self._sparkle: Dict[int, float] = {ch: 0.0 for ch in self.channel_ids}

        # Asymmetric envelope on kick strength (attack fast, release slower)
        self._kick_release_rate = 2.0  # units/sec (more momentum; less snappy)
        self._kick_gain = 2.0  # multiply kick signal before filtering (helps small flashes)

        # Color palettes (10 x 5 colors). We cycle palettes on special moments, and cycle colors on kicks.
        # Values are hues in [0,1) with a palette-specific saturation multiplier.
        self._palettes: List[Tuple[str, List[float], float]] = [
            # Higher-contrast, less pastel palettes (paired with ShowOptions.COLOR_VIBRANCE).
            # Hues are in [0,1). pal_sat is a palette-level multiplier; final saturation is also scaled by COLOR_VIBRANCE.
            ("neon_rush",   [0.84, 0.02, 0.12, 0.33, 0.58], 1.20),  # magenta, red, orange, green, electric blue
            ("laser_party", [0.92, 0.76, 0.10, 0.50, 0.62], 1.25),  # hot pink, violet, gold, cyan, blue
            ("acid_drop",   [0.18, 0.26, 0.40, 0.04, 0.86], 1.20),  # lime, chartreuse, teal, orange/red, pink
            ("ultraviolet", [0.72, 0.80, 0.88, 0.98, 0.58], 1.18),  # purple family + hot pink + blue
            ("embers",      [0.00, 0.03, 0.06, 0.10, 0.92], 1.10),  # red -> orange -> gold + pink accent
            ("deep_ice",    [0.52, 0.56, 0.60, 0.66, 0.12], 1.10),  # cyan -> blue -> indigo + gold accent
            ("toxic_sunset",[0.02, 0.08, 0.14, 0.28, 0.82], 1.15),  # red/orange/yellow -> green -> magenta
            ("club_rgb",    [0.00, 0.33, 0.66, 0.83, 0.12], 1.25),  # red, green, blue + magenta + orange
            ("high_contrast",[0.86, 0.58, 0.10, 0.34, 0.02], 1.22), # pink, blue, gold, green, red
            ("warm_mono",   [0.01, 0.04, 0.07, 0.10, 0.13], 1.05),  # warm monochrome, but more saturated
        ]
        self._palette_idx = 0
        self._color_idx = 0
        # Per-group hue assignments (kept stable between kicks; updated on kicks/reshuffles/palette changes)
        self._kick_hues: Dict[int, float] = {}
        self._high_hues: Dict[int, float] = {}
        self._hat_hues: Dict[int, float] = {}

        # Kick group selection (40% of lights for 2 kick beats, then reselect)
        self._kick_sel: set[int] = set()
        self._kick_beat_count = 0
        self._kick_sel_frac = 0.40
        # Default longer holds so roles feel stable (less random/flashy).
        self._kick_sel_hold_beats = int(getattr(ShowOptions, "GROUP_RESELECT_EVERY_BEATS", 8)) or 8

        # High-kick group selection (subset of remaining lights, same cadence)
        self._high_sel: set[int] = set()
        self._high_beat_count = 0
        self._high_sel_frac = 0.40
        self._high_sel_hold_beats = int(getattr(ShowOptions, "GROUP_RESELECT_EVERY_BEATS", 8)) or 8

        # Hat group selection (subset of remaining lights; shows hats/high-air)
        self._hat_sel: set[int] = set()
        self._hat_beat_count = 0
        self._hat_sel_frac = float(getattr(ShowOptions, "HAT_SEL_FRAC", 0.35))
        self._hat_sel_hold_beats = int(getattr(ShowOptions, "HAT_RESELECT_EVERY_BEATS", 8)) or 8

        # Now that palettes/indices exist, initialize groups (which assigns hues)
        self._select_kick_group(force=True)
        self._select_high_group(force=True)
        self._select_hat_group(force=True)

        # Double-time detection: if kicks become ~2x faster than recent, reshuffle immediately.
        self._kick_iv_hist: List[float] = []
        self._last_kick_t = 0.0

    @staticmethod
    def _median(xs: List[float]) -> float:
        if not xs:
            return 0.0
        ys = sorted(xs)
        m = len(ys) // 2
        if len(ys) % 2 == 1:
            return float(ys[m])
        return 0.5 * float(ys[m - 1] + ys[m])

    def step(self, f: Features, dt: float, now_s: float) -> Dict[int, Tuple[int,int,int]]:
        # Kick-band flash mode (simple): all lights flash from low kick frequencies (no BPM).
        energy = float(clamp((f.rms - 0.005) / 0.10, 0.0, 1.0))
        self.energy_ema = self.energy_ema * 0.85 + energy * 0.15
        tonal = float(clamp((0.35 - f.flatness) / 0.25, 0.0, 1.0))

        # Update tempo period estimate (used to scale beat animation)
        bpm = float(getattr(f, "bpm_est", 0.0) or 0.0)
        bpm_conf = float(getattr(f, "bpm_conf", 0.0) or 0.0)
        beat_active = bool(50.0 < bpm < 210.0 and bpm_conf >= 1.7)
        if beat_active:
            self._tempo_period_s = 0.90 * float(self._tempo_period_s) + 0.10 * float(60.0 / max(1e-6, bpm))

        # Decide which event triggers the "beat" animation.
        # If tempo grid is confident, prefer f.beat (stable) over f.kick (can chatter on basslines/hats).
        use_grid = bool(getattr(ShowOptions, "BEAT_USE_TEMPO_GRID", True))
        beat_evt = bool(getattr(f, "beat", False)) if (use_grid and beat_active) else bool(getattr(f, "kick", False))
        # Structural events (palette advance, group reshuffle, etc.) should *never* be driven by chattery kicks.
        # Use the tempo grid beat when available; otherwise fall back to kick.
        struct_evt = bool(getattr(f, "beat", False)) if (use_grid and beat_active) else bool(getattr(f, "kick", False))

        # ---------------------------------------------------------------------
        # Extra "show" events: buildup + drop + white sparkles (iLightShow-ish)
        # ---------------------------------------------------------------------
        riser = float(getattr(f, "riser", 0.0) or 0.0)
        onset_rate = float(getattr(f, "onset_rate", 0.0) or 0.0)
        dbl = bool(getattr(f, "double_burst", False))

        # Buildup mode: riser / rapid-onset / double-burst
        if bool(getattr(ShowOptions, "ENABLE_BUILDUP_EFFECTS", True)):
            thr = float(getattr(ShowOptions, "BUILDUP_RISER_THR", 0.55))
            hold = float(getattr(ShowOptions, "BUILDUP_HOLD_S", 1.4))
            # Only trust density-based buildup triggers when tempo is actually active/locked.
            if (riser >= thr) or (dbl and beat_active) or ((onset_rate >= 11.0) and beat_active):
                self._buildup_until = max(self._buildup_until, float(now_s) + hold + 1.0 * riser)
        buildup = bool(float(now_s) < float(self._buildup_until))

        # Drop detection: riser was high and then falls quickly -> flash + palette jump
        if bool(getattr(ShowOptions, "ENABLE_DROP_FLASH", True)):
            if bool(getattr(ShowOptions, "DROP_MIN_BEAT_ACTIVE", True)) and (not beat_active):
                # Don't fire drop flashes when beat isn't locked (feels random).
                pass
            else:
                peak_thr = float(getattr(ShowOptions, "DROP_RISER_PEAK_THR", 0.72))
                fall_thr = float(getattr(ShowOptions, "DROP_RISER_FALL_THR", 0.33))
                if (self._last_riser >= peak_thr) and (riser <= fall_thr) and beat_evt:
                    self._drop_env = 1.0
                    if bool(getattr(ShowOptions, "DROP_PALETTE_JUMP", True)) and ShowOptions.ENABLE_PALETTES:
                        # jump palette for a "drop hit" feel
                        self._palette_idx = (self._palette_idx + 1 + self._rng.randrange(3)) % len(self._palettes)
                        self._update_group_hues()

        # Decay drop flash envelope (white mix applied in render loop)
        if self._drop_env > 1e-4:
            ddec = float(getattr(ShowOptions, "DROP_FLASH_DECAY_S", 0.25))
            self._drop_env *= float(math.exp(-float(dt) / max(1e-6, ddec)))
        else:
            self._drop_env = 0.0

        # White sparkles during buildup (random lights flash white briefly)
        if bool(getattr(ShowOptions, "ENABLE_WHITE_SPARKLES", True)):
            # decay existing sparkles
            sdec = float(getattr(ShowOptions, "BUILDUP_SPARKLE_DECAY_S", 0.12))
            if sdec > 0:
                kdec = float(math.exp(-float(dt) / max(1e-6, sdec)))
                for ch in self._sparkle:
                    self._sparkle[ch] *= kdec
            else:
                for ch in self._sparkle:
                    self._sparkle[ch] = 0.0

            if buildup and (self.n > 0):
                base_rate = float(getattr(ShowOptions, "BUILDUP_SPARKLES_PER_S", 7.0))
                # more riser + more onset density -> more sparkles
                rate = base_rate * (0.35 + 1.6 * riser + 0.10 * min(20.0, onset_rate))
                # dt-based Poisson-ish spawn; cap probability for stability
                p = float(clamp(rate * float(dt), 0.0, 0.85))
                if self._rng.random() < p:
                    # Prefer non-kick-group lights so the kick group stays readable
                    pool = [ch for ch in self.channel_ids if ch not in self._kick_sel] or list(self.channel_ids)
                    ch = int(self._rng.choice(pool))
                    self._sparkle[ch] = max(self._sparkle.get(ch, 0.0), 1.0)

        # Track last riser for drop detection
        self._last_riser = float(riser)

        # Flash envelope from kick band strength (asymmetric: fast up, slower down)
        if ShowOptions.ENABLE_KICKS:
            ks = float(getattr(f, "kick_strength", 0.0))
            if f.kick:
                ks = 1.0
            if ks >= self._beat_pulse:
                self._beat_pulse = ks
            else:
                self._beat_pulse = max(0.0, self._beat_pulse - float(dt) * float(self._kick_release_rate))
        else:
            self._beat_pulse = 0.0

        if struct_evt and ShowOptions.ENABLE_KICKS:
            # Double-time detection (interval suddenly halves)
            if self._last_kick_t > 0.0:
                dk = float(now_s - self._last_kick_t)
                if 0.06 < dk < 1.0:
                    if self._kick_iv_hist:
                        med = self._median(self._kick_iv_hist)
                        if med > 0.0 and dk < med * 0.65:
                            # sudden ~2x tempo -> reshuffle immediately
                            self._select_kick_group(force=True)
                            self._kick_beat_count = 0
                            self._select_high_group(force=True)
                            self._high_beat_count = 0
                            # also jump palette on double-time moments (feels hype)
                            if ShowOptions.ENABLE_PALETTES and ShowOptions.ENABLE_PALETTE_SWITCH:
                                self._palette_idx = (self._palette_idx + 1 + self._rng.randrange(3)) % len(self._palettes)
                    self._kick_iv_hist.append(dk)
                    if len(self._kick_iv_hist) > 10:
                        self._kick_iv_hist.pop(0)
            self._last_kick_t = float(now_s)

            self._kick_beat_count += 1
            if (self._kick_beat_count - 1) % max(1, int(self._kick_sel_hold_beats)) == 0:
                self._select_kick_group(force=True)
                self._select_high_group(force=True)

            self._high_beat_count += 1
            if (self._high_beat_count - 1) % max(1, int(self._high_sel_hold_beats)) == 0:
                self._select_high_group(force=True)

            self._hat_beat_count += 1
            if (self._hat_beat_count - 1) % max(1, int(self._hat_sel_hold_beats)) == 0:
                self._select_hat_group(force=True)

            # (Palette/color changes moved below to a slower beat cadence to reduce flashiness.)

        # Decide "drop" mode if sustained energy + bass presence
        if self.energy_ema > 0.55 and f.bass > 0.38:
            self.mode_hold = min(1.0, self.mode_hold + dt * 0.9)
        else:
            self.mode_hold = max(0.0, self.mode_hold - dt * 0.6)
        self.mode = "chill"

        # Kick triggers
        # Keep "kick triggers" limited to true kick detections so it still feels percussive,
        # but avoid using them for structural show changes (handled by struct_evt above).
        if bool(getattr(f, "kick", False)):
            self.strobe = 1.0
            self.last_kick_time = now_s
            # kick tempo tracking + double-time detection
            if self._prev_kick_time > 0.0:
                dk = now_s - self._prev_kick_time
                # keep plausible kick intervals
                if 0.12 < dk < 1.25:
                    # Robust BPM tracking:
                    # - keep a rolling median of recent intervals
                    # - ignore very long dk outliers (often caused by missed kick detections)
                    #   so BPM doesn't suddenly halve during quiet sections.
                    med = self._median(self._kick_intervals) if self._kick_intervals else 0.0

                    # Outlier gating once we have a baseline
                    accept = True
                    if med > 0.0:
                        # If dk is much longer than baseline, assume missed beats; don't update BPM.
                        if dk > med * 1.85:
                            accept = False

                    if accept:
                        self._kick_intervals.append(dk)
                        if len(self._kick_intervals) > 10:
                            self._kick_intervals.pop(0)
                        med2 = self._median(self._kick_intervals)
                        if med2 > 0.0:
                            self.kick_bpm = 60.0 / med2

                        # "double time" = interval drops sharply vs recent median (tempo jumps up)
                        # Require it to happen twice close together to reduce false positives.
                        if (med2 > 0.25) and (dk < med2 * 0.65):
                            self._double_time_hits += 1
                        else:
                            self._double_time_hits = max(0, self._double_time_hits - 1)

                        if self._double_time_hits >= 2:
                            self._double_time_until = now_s + 3.0
                            self._double_time_hits = 0

            self._prev_kick_time = now_s
            # palette jump on kicks (more in drop mode)
            self.base_h = (self.base_h + (0.08 if self.mode == "chill" else 0.16)) % 1.0
            # advance a "chaser" index so different lights take turns peaking
            if self.n > 0:
                self.chase_idx = (self.chase_idx + 1) % self.n
            # occasional drop flash
            if self.mode == "drop" and f.bass > 0.45:
                self.drop_flash = max(self.drop_flash, 0.9)

        # Riser drives hue spin + brightness ramp
        # Hue drift gives motion even when music is steady; during buildup, drift speeds up.
        if bool(getattr(ShowOptions, "ENABLE_HUE_DRIFT", True)):
            drift = float(getattr(ShowOptions, "DRIFT_RATE", 0.06))
            boost = float(getattr(ShowOptions, "BUILDUP_DRIFT_BOOST", 2.6)) if buildup else 1.0
            self.h_spin = (self.h_spin + float(dt) * float(drift) * float(boost)) % 1.0
        else:
            self.h_spin = (self.h_spin + dt * (0.10 + 0.8 * riser)) % 1.0

        # Decays
        self.strobe = max(0.0, self.strobe - dt * (8.0 if self.mode == "drop" else 5.0))
        self.drop_flash = max(0.0, self.drop_flash - dt * 4.5)
        self.beat_flash = max(0.0, self.beat_flash - dt * 8.5)
        # (beat_pulse handled by asymmetric envelope above)
        self._mel_pulse = 0.0

        # Trigger + run beat brightness animation
        if bool(getattr(ShowOptions, "BEAT_ANIM_ENABLED", True)):
            if beat_evt:
                retrig_frac = float(getattr(ShowOptions, "BEAT_ANIM_RETRIGGER_MIN_FRAC", 0.35))
                retrig_min_s = float(getattr(ShowOptions, "BEAT_ANIM_RETRIGGER_MIN_S", 0.09))
                retrig_s = max(retrig_min_s, retrig_frac * float(self._tempo_period_s))
                if (now_s - float(self._last_beat_trigger_t)) >= retrig_s:
                    self._beat_anim = 1.0
                    self._last_beat_trigger_t = float(now_s)

            decay_frac = float(getattr(ShowOptions, "BEAT_ANIM_DECAY_FRAC", 0.55))
            decay_s = float(self._tempo_period_s) * decay_frac
            decay_s = float(clamp(decay_s,
                                 float(getattr(ShowOptions, "BEAT_ANIM_DECAY_MIN_S", 0.10)),
                                 float(getattr(ShowOptions, "BEAT_ANIM_DECAY_MAX_S", 0.55))))
            self._beat_anim_decay_s = decay_s
            if self._beat_anim > 1e-4 and decay_s > 1e-6:
                # exponential decay feels "polished" vs linear ramp-down
                self._beat_anim *= float(math.exp(-float(dt) / float(decay_s)))
            else:
                self._beat_anim = 0.0
        else:
            self._beat_anim = 0.0

        # Slower palette cadence: advance color on a beat grid, not every kick.
        if ShowOptions.ENABLE_PALETTES:
            adv_every = max(1, int(getattr(ShowOptions, "COLOR_ADVANCE_EVERY_BEATS", 4)))
            # Prefer tempo grid beats if active; otherwise fall back to kick count.
            if (beat_active and bool(getattr(f, "beat", False))):
                if (int(getattr(f, "beat_idx", 0)) % adv_every) == 0:
                    self._color_idx += 1
                    self._update_group_hues()
            elif bool(getattr(f, "kick", False)):
                if (int(self._kick_beat_count) % adv_every) == 0:
                    self._color_idx += 1
                    self._update_group_hues()
            # Occasionally switch palette even more slowly.
            if ShowOptions.ENABLE_PALETTE_SWITCH:
                pal_every = adv_every * 8
                pal_every = max(16, int(pal_every))
                if (beat_active and bool(getattr(f, "beat", False)) and (int(getattr(f, "beat_idx", 0)) % pal_every) == 0):
                    self._palette_idx = (self._palette_idx + 1) % len(self._palettes)
                    self._update_group_hues()
                elif (not beat_active) and bool(getattr(f, "kick", False)) and (int(self._kick_beat_count) % pal_every) == 0:
                    self._palette_idx = (self._palette_idx + 1) % len(self._palettes)
                    self._update_group_hues()

        # Compute global brightness/saturation: calm baseline + strong kick flashes
        # Slightly brighter overall (per user feedback).
        base_v = 0.07 + 0.22 * self.energy_ema
        base_s = 0.10 + 0.25 * tonal

        # No white strobing in beat-only mode.

        # No drop flash in beat-only mode.

        # Otherwise: same color for all channels; intensity is the show.
        out: Dict[int, Tuple[int,int,int]] = {}
        pal_name, pal_hues, pal_sat = self._palettes[self._palette_idx]
        if ShowOptions.ENABLE_PALETTES:
            # Base saturation is low-ish for "warm white" feel; palette raises it into color.
            vib = float(getattr(ShowOptions, "COLOR_VIBRANCE", 1.35))
            s = clamp(base_s * float(pal_sat) * vib, 0.0, 1.0)
        else:
            s = clamp(base_s * 0.30, 0.0, 1.0)
        v_base = clamp(base_v, 0.0, 1.0)
        # Main "kick group" brightness:
        # Use beat animation to guarantee a visible peak, and optionally add some extra punch from kick strength.
        min_peak = float(getattr(ShowOptions, "BEAT_ANIM_MIN_PEAK", 0.65))
        max_peak = float(getattr(ShowOptions, "BEAT_ANIM_MAX_PEAK", 1.00))
        # Drive uses energy + kick_strength (when present) so big drops still look bigger.
        ks = float(getattr(f, "kick_strength", 0.0) or 0.0)
        drive = clamp(0.45 * float(self.energy_ema) + 0.55 * float(ks), 0.0, 1.0)
        peak = clamp(min_peak + (max_peak - min_peak) * drive, 0.0, 1.0)
        v_flash = clamp(v_base + peak * float(self._beat_anim), 0.0, 1.0)

        # Non-kick lights respond to kick click + hats using the same smoothing model as kicks:
        # - attack: instant
        # - release: rate-limited
        # - output: low-pass filtered (same RC as main intensity)
        high_raw = float(getattr(f, "high_strength", 0.0)) if ShowOptions.ENABLE_HIGH else 0.0

        # High band is very spiky; apply a slew-limited attack + slower release to avoid "instant max".
        if high_raw >= self._high_pulse:
            rise = float(dt) * float(self._high_attack_rate)
            self._high_pulse = min(float(high_raw), float(self._high_pulse) + rise)
        else:
            self._high_pulse = max(0.0, float(self._high_pulse) - float(dt) * float(self._high_release_rate))

        self._high_filt.dt = float(dt)
        self._high_filt.update_alpha(max(1e-4, float(self._high_lp_rc)))
        high_lp = float(self._high_filt.update(self._high_pulse))

        # Hat envelope (very high transients)
        hat_raw = float(getattr(f, "hat_strength", 0.0)) if bool(getattr(ShowOptions, "ENABLE_HATS", True)) else 0.0
        if hat_raw >= self._hat_pulse:
            rise = float(dt) * float(self._hat_attack_rate)
            self._hat_pulse = min(float(hat_raw), float(self._hat_pulse) + rise)
        else:
            self._hat_pulse = max(0.0, float(self._hat_pulse) - float(dt) * float(self._hat_release_rate))

        self._hat_filt.dt = float(dt)
        self._hat_filt.update_alpha(max(1e-4, float(self._hat_lp_rc)))
        hat_lp = float(self._hat_filt.update(self._hat_pulse))

        for ch in self.channel_ids:
            in_kick_group = (ch in self._kick_sel)
            if in_kick_group and ShowOptions.ENABLE_KICKS:
                h = float(self._kick_hues.get(ch, pal_hues[0] if pal_hues else 0.10)) if ShowOptions.ENABLE_PALETTES else 0.10
                # slight hue motion so it doesn't feel static between palette changes
                if bool(getattr(ShowOptions, "ENABLE_HUE_DRIFT", True)):
                    h = (h + 0.06 * float(self.h_spin)) % 1.0
                r, g, b = hsv_to_rgb(h, s, v_flash)
                # White mixing (sparkles / drop)
                w = 0.0
                if self._drop_env > 0.0 and bool(getattr(ShowOptions, "ENABLE_DROP_FLASH", True)):
                    w = max(w, float(getattr(ShowOptions, "DROP_WHITE", 0.85)) * float(self._drop_env))
                if buildup and bool(getattr(ShowOptions, "ENABLE_WHITE_SPARKLES", True)):
                    w = max(w, float(getattr(ShowOptions, "BUILDUP_WHITE_MAX", 0.95)) * float(self._sparkle.get(ch, 0.0)))
                if w > 1e-4:
                    w = float(clamp(w, 0.0, 1.0))
                    ww = float(v_flash)
                    r = (1.0 - w) * float(r) + w * ww
                    g = (1.0 - w) * float(g) + w * ww
                    b = (1.0 - w) * float(b) + w * ww
                out[ch] = rgb_to_u16(r, g, b, gamma=True)
            else:
                if not ShowOptions.ENABLE_BACKGROUND and (not ShowOptions.ENABLE_HIGH):
                    out[ch] = (0, 0, 0)
                else:
                    in_high = (ch in self._high_sel)
                    in_hat = (ch in self._hat_sel)
                    # Other lights: respond to kick click + hats with brighter/snappier colors.
                    h_high = float(self._high_hues.get(ch, pal_hues[2] if len(pal_hues) > 2 else (pal_hues[0] if pal_hues else 0.62))) if ShowOptions.ENABLE_PALETTES else 0.62
                    h_hat = float(self._hat_hues.get(ch, pal_hues[1] if len(pal_hues) > 1 else (pal_hues[0] if pal_hues else 0.12))) if ShowOptions.ENABLE_PALETTES else 0.12

                    v_pulse = (0.55 * float(getattr(ShowOptions, "HIGH_GAIN", 1.0)) * high_lp) if (in_high and ShowOptions.ENABLE_HIGH) else 0.0
                    v_other = clamp(v_base + v_pulse, 0.0, 1.0)
                    if in_hat:
                        v_other = clamp(v_other + (0.38 * float(getattr(ShowOptions, "HAT_GAIN", 2.2)) * hat_lp), 0.0, 1.0)
                    if ShowOptions.ENABLE_BACKGROUND:
                        v_other = max(v_other, v_base * 0.9)
                    else:
                        v_other = max(v_other, 0.0)

                    # During buildup, spin hues faster on high group for that "riser rainbow" feel.
                    h2 = h_high
                    if bool(getattr(ShowOptions, "ENABLE_HUE_DRIFT", True)):
                        spin = float(self.h_spin)
                        if buildup and in_high:
                            h2 = (h2 + 0.22 * spin + 0.10 * float(getattr(f, "beat_phase", 0.0))) % 1.0
                        else:
                            h2 = (h2 + 0.08 * spin) % 1.0
                    if in_hat:
                        # Hat group anchors to its own hue and stays crisp.
                        h2 = (h_hat + 0.06 * float(self.h_spin)) % 1.0
                    s_boost = (0.30 + 0.60 * min(1.0, float(getattr(ShowOptions, "HIGH_GAIN", 1.0)) * high_lp)) if in_high else 0.20
                    if in_hat:
                        s_boost = max(s_boost, 0.55 + 0.25 * min(1.0, float(getattr(ShowOptions, "HAT_GAIN", 2.2)) * hat_lp))
                    s2 = clamp(s * s_boost, 0.0, 1.0)
                    r2, g2, b2 = hsv_to_rgb(h2, s2, v_other)
                    # White mixing for non-kick lights too (sparkles / drop)
                    w2 = 0.0
                    if self._drop_env > 0.0 and bool(getattr(ShowOptions, "ENABLE_DROP_FLASH", True)):
                        w2 = max(w2, float(getattr(ShowOptions, "DROP_WHITE", 0.85)) * float(self._drop_env))
                    if buildup and bool(getattr(ShowOptions, "ENABLE_WHITE_SPARKLES", True)):
                        w2 = max(w2, float(getattr(ShowOptions, "BUILDUP_WHITE_MAX", 0.95)) * float(self._sparkle.get(ch, 0.0)))
                    if in_hat:
                        w2 = max(w2, float(getattr(ShowOptions, "HAT_WHITE_TICK", 0.12)) * float(hat_lp))
                    if w2 > 1e-4:
                        w2 = float(clamp(w2, 0.0, 1.0))
                        ww2 = float(v_other)
                        r2 = (1.0 - w2) * float(r2) + w2 * ww2
                        g2 = (1.0 - w2) * float(g2) + w2 * ww2
                        b2 = (1.0 - w2) * float(b2) + w2 * ww2
                    out[ch] = rgb_to_u16(r2, g2, b2, gamma=True)

        return out

    def _update_group_hues(self) -> None:
        """
        Assign palette hues to each member of kick/high groups without duplicates when possible.
        If a group has more members than palette colors, we keep hues unique by adding a tiny offset per wrap.
        """
        try:
            _, pal_hues, _ = self._palettes[self._palette_idx]
        except Exception:
            pal_hues = [0.10]
        if not pal_hues:
            pal_hues = [0.10]

        def wrap_half(x: float) -> float:
            return ((x + 0.5) % 1.0) - 0.5

        def circ_mean(hues: List[float]) -> float:
            if not hues:
                return 0.0
            ang = np.array([float(h) % 1.0 for h in hues], dtype=np.float64) * (2.0 * math.pi)
            c = float(np.mean(np.cos(ang)))
            s = float(np.mean(np.sin(ang)))
            if (c == 0.0) and (s == 0.0):
                return float(hues[0]) % 1.0
            return (math.atan2(s, c) / (2.0 * math.pi)) % 1.0

        def circ_dist(a: float, b: float) -> float:
            d = abs((a - b) % 1.0)
            return min(d, 1.0 - d)

        def spread_palette(hues_in: List[float]) -> List[float]:
            """
            Spread palette hues away from their circular mean and enforce a minimum separation.
            This makes "unique" hues look less like the same shade in practice.
            """
            if not hues_in:
                return [0.10]
            center = circ_mean(hues_in)
            factor = float(ShowOptions.PALETTE_HUE_SPREAD)
            min_sep = float(ShowOptions.PALETTE_MIN_HUE_SEP)

            out = [(center + wrap_half(((float(h) % 1.0) - center) * factor)) % 1.0 for h in hues_in]
            out = sorted([float(h) % 1.0 for h in out])
            # relax a few passes
            for _ in range(6):
                changed = False
                for i in range(1, len(out)):
                    if circ_dist(out[i], out[i - 1]) < min_sep:
                        out[i] = (out[i] + (min_sep - circ_dist(out[i], out[i - 1]) + 1e-3)) % 1.0
                        changed = True
                out = sorted(out)
                if not changed:
                    break
            return out

        def assign(chs: List[int], seed: int) -> Dict[int, float]:
            if not chs:
                return {}
            rng = random.Random(seed)
            chs2 = list(chs)
            rng.shuffle(chs2)
            hues = spread_palette(list(pal_hues))
            rng.shuffle(hues)
            out: Dict[int, float] = {}
            for i, ch in enumerate(chs2):
                base = float(hues[i % len(hues)])
                wrap = i // len(hues)
                # small hue shift per wrap to keep uniqueness if group > palette size
                h = (base + float(ShowOptions.PALETTE_WRAP_OFFSET) * wrap) % 1.0
                out[ch] = h
            return out

        self._kick_hues = assign(sorted(self._kick_sel), seed=1000 + self._palette_idx * 97 + self._color_idx * 7 + self._kick_beat_count * 3)
        self._high_hues = assign(sorted(self._high_sel), seed=2000 + self._palette_idx * 97 + self._color_idx * 11 + self._high_beat_count * 3)
        self._hat_hues = assign(sorted(self._hat_sel), seed=3000 + self._palette_idx * 97 + self._color_idx * 13 + self._hat_beat_count * 3)

    def _select_kick_group(self, force: bool = False) -> None:
        if self.n <= 0:
            self._kick_sel = set()
            return
        if (not force) and self._kick_sel:
            return
        k = max(1, int(round(self.n * float(self._kick_sel_frac))))
        k = min(self.n, k)
        # Use rng from instance to keep deterministic-ish behavior across runs
        self._kick_sel = set(self._rng.sample(self.channel_ids, k))
        self._update_group_hues()

    def _select_high_group(self, force: bool = False) -> None:
        if self.n <= 0:
            self._high_sel = set()
            return
        if (not force) and self._high_sel:
            return
        # Prefer choosing from non-kick-group lights so roles don't overlap.
        if ShowOptions.ENABLE_KICKS:
            pool = [ch for ch in self.channel_ids if ch not in self._kick_sel]
        else:
            pool = list(self.channel_ids)
        if not pool:
            pool = list(self.channel_ids)
        k = max(1, int(round(len(pool) * float(self._high_sel_frac))))
        k = min(len(pool), k)
        self._high_sel = set(self._rng.sample(pool, k))
        self._update_group_hues()

    def _select_hat_group(self, force: bool = False) -> None:
        if self.n <= 0:
            self._hat_sel = set()
            return
        if (not force) and self._hat_sel:
            return
        # Prefer choosing from non-kick and non-high lights so roles don't overlap.
        pool = [ch for ch in self.channel_ids if (ch not in self._kick_sel) and (ch not in self._high_sel)]
        if not pool:
            pool = [ch for ch in self.channel_ids if ch not in self._kick_sel] or list(self.channel_ids)
        k = max(1, int(round(len(pool) * float(self._hat_sel_frac))))
        k = min(len(pool), k)
        self._hat_sel = set(self._rng.sample(pool, k))
        self._update_group_hues()

    def _reshuffle_groups(self, force: bool = False) -> None:
        if self.n <= 0:
            return
        if (not force) and self._group_role:
            return
        ids = list(self.channel_ids)
        self._rng.shuffle(ids)
        n = len(ids)
        n_kick = max(1, int(round(n * 0.50)))
        n_hat = max(1, int(round(n * 0.25)))
        # ensure total <= n
        if n_kick + n_hat > n:
            n_hat = max(0, n - n_kick)
        n_mel = max(0, n - n_kick - n_hat)
        roles = (["kick"] * n_kick) + (["hat"] * n_hat) + (["mel"] * n_mel)
        # pad/truncate
        roles = (roles + ["mel"] * n)[:n]
        self._group_role = {ch: roles[i] for i, ch in enumerate(ids)}


# -----------------------------
# Audio IO + Ring Buffer
# -----------------------------

class AudioRing:
    def __init__(self, max_samples: int) -> None:
        self.buf = np.zeros(max_samples, dtype=np.float32)
        self.max = max_samples
        self.w = 0
        self.lock = threading.Lock()
        self.filled = False

    def push(self, x: np.ndarray) -> None:
        x = x.astype(np.float32, copy=False).reshape(-1)
        n = x.size
        if n <= 0:
            return
        with self.lock:
            if n >= self.max:
                self.buf[:] = x[-self.max:]
                self.w = 0
                self.filled = True
                return
            end = self.w + n
            if end <= self.max:
                self.buf[self.w:end] = x
            else:
                k = self.max - self.w
                self.buf[self.w:] = x[:k]
                self.buf[: end - self.max] = x[k:]
            self.w = end % self.max
            if self.w == 0:
                self.filled = True

    def read_last(self, n: int) -> np.ndarray:
        n = min(n, self.max)
        with self.lock:
            if not self.filled and self.w < n:
                # not enough yet
                return self.buf[:self.w].copy()
            start = (self.w - n) % self.max
            if start < self.w:
                return self.buf[start:self.w].copy()
            return np.concatenate([self.buf[start:], self.buf[:self.w]]).copy()

    def read_last_offset(self, n: int, end_offset: int) -> np.ndarray:
        """
        Read the last `n` samples ending `end_offset` samples before the current write head.
        Useful for adding a positive audio->lights delay (e.g. Bluetooth latency compensation).
        """
        n = min(int(n), self.max)
        end_offset = max(0, int(end_offset))
        end_offset = min(end_offset, self.max - 1)
        with self.lock:
            end = (self.w - end_offset) % self.max
            if not self.filled:
                # available samples are [0, self.w)
                if end_offset >= self.w:
                    return np.array([], dtype=np.float32)
                if self.w < n + end_offset:
                    # not enough yet
                    return self.buf[: max(0, self.w - end_offset)].copy()

            start = (end - n) % self.max
            if start < end:
                return self.buf[start:end].copy()
            return np.concatenate([self.buf[start:], self.buf[:end]]).copy()


# -----------------------------
# Debug Preview UI (pygame)
# -----------------------------

class PreviewWindow:
    """
    Simple pygame window that shows the exact per-channel RGB colors being sent to the Hue bridge.
    One tile per channel_id. Intended as a debug "mirror" preview.
    """

    def __init__(
        self,
        channel_ids: List[int],
        scale: int = DEFAULT_PREVIEW_SCALE,
        hz: float = DEFAULT_PREVIEW_HZ,
        layout: str = DEFAULT_PREVIEW_LAYOUT,
        title: str = "Hue Sync Preview",
        show_spectrum: bool = False,
        spectrum_height: int = DEFAULT_SPECTRUM_HEIGHT,
        spectrum_max_hz: float = DEFAULT_SPECTRUM_MAX_HZ,
        window_width: int = 0,
    ) -> None:
        if pygame is None:
            raise RuntimeError("pygame is not installed (pip install pygame) but --preview was requested.")
        self.channel_ids = list(channel_ids)
        self.n = len(self.channel_ids)
        self.scale = int(scale)
        self.hz = float(hz)
        self.layout = str(layout or "grid").lower()
        self.title = title
        self.show_spectrum = bool(show_spectrum)
        self.spectrum_h = int(spectrum_height)
        self.spectrum_max_hz = float(spectrum_max_hz)
        self.window_width = int(window_width)

        self._lock = threading.Lock()
        self._frame_rgb8: Dict[int, Tuple[int, int, int]] = {ch: (0, 0, 0) for ch in self.channel_ids}
        self._frame_v: Dict[int, float] = {ch: 0.0 for ch in self.channel_ids}
        # Per-channel role labels (for debugging group selection)
        # "" = none, "K" = kick group, "H" = high group, "K+H" = both (should be rare).
        self._role_by_ch: Dict[int, str] = {ch: "" for ch in self.channel_ids}
        self._spec_freqs: Optional[np.ndarray] = None
        self._spec_mag: Optional[np.ndarray] = None
        self._kick_lo = 40.0
        self._kick_hi = 130.0
        self._high_lo = 1200.0
        self._high_hi = 12000.0
        self._hat_lo = 8000.0
        self._hat_hi = 16000.0
        # Debug signals (for visualizing analyzer/beat internals)
        self._dbg_onset = 0.0
        self._dbg_onset_env = 0.0
        self._dbg_onset_thr = 0.0
        self._dbg_onset_evt = False
        self._dbg_beat = False
        self._dbg_kick = False
        self._dbg_bpm = 0.0
        self._dbg_bpm_conf = 0.0
        self._dbg_beat_active = False
        self._dbg_beat_phase = 0.0
        self._dbg_onset_src = "mix"
        self._dbg_buildup = False
        self._dbg_drop = 0.0
        self._dbg_bass_dz = 0.0
        self._dbg_bass_db_z = 0.0
        self._dbg_db_diff = 0.0
        self._dbg_flux_dev = 0.0
        self._dbg_flux_z = 0.0
        self._stop = threading.Event()

        if self.n <= 0:
            self.cols, self.rows = 1, 1
        elif self.layout == "circle":
            self.cols, self.rows = 1, 1
        else:
            self.cols = int(math.ceil(math.sqrt(self.n)))
            self.rows = int(math.ceil(self.n / max(1, self.cols)))

        tiles_w = max(240, self.cols * self.scale)
        tiles_h = max(180, self.rows * self.scale)
        # Keep tile sizes fixed; optionally expand total window width so spectrum can use more horizontal resolution.
        if self.show_spectrum:
            target_w = max(tiles_w, 1200)  # sensible default; wide enough to read spectrum
        else:
            target_w = tiles_w
        if self.window_width > 0:
            target_w = max(target_w, self.window_width)

        self._tiles_w = tiles_w
        self._tiles_h = tiles_h
        self._dbg_h = 86 if self.show_spectrum else 0
        self.w = target_w
        self.h = tiles_h + self._dbg_h + (self.spectrum_h if self.show_spectrum else 0)

    def stop(self) -> None:
        self._stop.set()

    def update_from_rgb16(self, rgb16_by_channel: Dict[int, Tuple[int, int, int]]) -> None:
        # Convert u16 (0..65535) to u8 (0..255) for preview.
        with self._lock:
            for ch in self.channel_ids:
                r16, g16, b16 = rgb16_by_channel.get(ch, (0, 0, 0))
                r8, g8, b8 = (int(r16) >> 8, int(g16) >> 8, int(b16) >> 8)
                self._frame_rgb8[ch] = (r8, g8, b8)
                self._frame_v[ch] = float(max(r8, g8, b8) / 255.0)

    def update_roles(self, kick_group: set[int], high_group: set[int], hat_group: Optional[set[int]] = None) -> None:
        """
        Provide the current per-channel group membership (from the show engine).
        """
        with self._lock:
            hat_group = hat_group or set()
            for ch in self.channel_ids:
                in_k = ch in kick_group
                in_h = ch in high_group
                in_t = ch in hat_group
                parts = []
                if in_k:
                    parts.append("K")
                if in_h:
                    parts.append("H")
                if in_t:
                    parts.append("T")
                self._role_by_ch[ch] = "+".join(parts)

    def update_spectrum(
        self,
        freqs: np.ndarray,
        mag_raw: np.ndarray,
        kick_lo: float,
        kick_hi: float,
        high_lo: float,
        high_hi: float,
        hat_lo: float,
        hat_hi: float,
        onset: float,
        onset_env: float,
        onset_thr: float,
        onset_evt: bool,
        beat: bool,
        kick: bool,
        bpm: float,
        bpm_conf: float,
        beat_active: bool,
        beat_phase: float,
        onset_src: str,
        buildup: bool,
        drop_env: float,
        bass_dz: float,
        bass_db_z: float,
        db_diff: float,
        flux_dev: float,
        flux_z: float,
    ) -> None:
        with self._lock:
            self._spec_freqs = freqs.astype(np.float32, copy=False)
            self._spec_mag = mag_raw.astype(np.float32, copy=False)
            self._kick_lo = float(kick_lo)
            self._kick_hi = float(kick_hi)
            self._high_lo = float(high_lo)
            self._high_hi = float(high_hi)
            self._hat_lo = float(hat_lo)
            self._hat_hi = float(hat_hi)
            self._dbg_onset = float(onset)
            self._dbg_onset_env = float(onset_env)
            self._dbg_onset_thr = float(onset_thr)
            self._dbg_onset_evt = bool(onset_evt)
            self._dbg_beat = bool(beat)
            self._dbg_kick = bool(kick)
            self._dbg_bpm = float(bpm)
            self._dbg_bpm_conf = float(bpm_conf)
            self._dbg_beat_active = bool(beat_active)
            self._dbg_beat_phase = float(beat_phase)
            self._dbg_onset_src = str(onset_src or "")
            self._dbg_buildup = bool(buildup)
            self._dbg_drop = float(drop_env)
            self._dbg_bass_dz = float(bass_dz)
            self._dbg_bass_db_z = float(bass_db_z)
            self._dbg_db_diff = float(db_diff)
            self._dbg_flux_dev = float(flux_dev)
            self._dbg_flux_z = float(flux_z)

    def set_status_line(self, text: str) -> None:
        # Optional one-liner for things like audio delay / latency notes.
        with self._lock:
            self._status_line = str(text or "")

    def loop(self, on_quit: Optional[callable] = None) -> None:
        pygame.init()
        pygame.display.set_caption(self.title)
        screen = pygame.display.set_mode((self.w, self.h))
        clock = pygame.time.Clock()
        font = None
        try:
            font = pygame.font.SysFont("monospace", max(12, int(self.scale * 0.14)))
        except Exception:
            font = None

        while not self._stop.is_set():
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self._stop.set()
                    if on_quit:
                        try:
                            on_quit()
                        except Exception:
                            pass
                    break

            screen.fill((10, 10, 12))
            with self._lock:
                items = [
                    (
                        ch,
                        self._frame_rgb8.get(ch, (0, 0, 0)),
                        float(self._frame_v.get(ch, 0.0)),
                        str(self._role_by_ch.get(ch, "")),
                    )
                    for ch in self.channel_ids
                ]
                freqs = None if self._spec_freqs is None else self._spec_freqs.copy()
                mag = None if self._spec_mag is None else self._spec_mag.copy()
                klo, khi = self._kick_lo, self._kick_hi
                hlo, hhi = self._high_lo, self._high_hi
                tlo, thi = self._hat_lo, self._hat_hi
                dbg_onset = float(self._dbg_onset)
                dbg_onset_env = float(self._dbg_onset_env)
                dbg_onset_thr = float(self._dbg_onset_thr)
                dbg_onset_evt = bool(self._dbg_onset_evt)
                dbg_beat = bool(self._dbg_beat)
                dbg_kick = bool(self._dbg_kick)
                dbg_bpm = float(self._dbg_bpm)
                dbg_bpm_conf = float(self._dbg_bpm_conf)
                dbg_beat_active = bool(self._dbg_beat_active)
                dbg_beat_phase = float(self._dbg_beat_phase)
                dbg_onset_src = str(getattr(self, "_dbg_onset_src", ""))
                dbg_buildup = bool(getattr(self, "_dbg_buildup", False))
                dbg_drop = float(getattr(self, "_dbg_drop", 0.0))
                dbg_bass_dz = float(self._dbg_bass_dz)
                dbg_bass_db_z = float(self._dbg_bass_db_z)
                dbg_db_diff = float(self._dbg_db_diff)
                dbg_flux_dev = float(self._dbg_flux_dev)
                dbg_flux_z = float(self._dbg_flux_z)
                status_line = str(getattr(self, "_status_line", ""))

            # Layout:
            # - tiles at y=[0, tiles_h)
            # - debug strip at y=[tiles_h, tiles_h+dbg_h) (only when show_spectrum)
            # - spectrum at bottom
            tile_area_h = self._tiles_h
            tile_x_off = int((self.w - self._tiles_w) // 2) if self.w > self._tiles_w else 0

            if self.layout == "circle" and self.n > 0:
                cx, cy = self.w // 2, tile_area_h // 2
                radius = int(min(self.w, self.h) * 0.35)
                box = max(16, int(self.scale * 0.55))
                for i, (ch, rgb, v, role) in enumerate(items):
                    ang = (2.0 * math.pi * i) / max(1, self.n)
                    x = int(cx + radius * math.cos(ang) - box / 2)
                    y = int(cy + radius * math.sin(ang) - box / 2)
                    pygame.draw.rect(screen, rgb, pygame.Rect(x, y, box, box))
                    pygame.draw.rect(screen, (30, 30, 34), pygame.Rect(x, y, box, box), 2)
                    if font:
                        fg = (10, 10, 10) if v > 0.65 else (240, 240, 240)
                        tag = f"[{role}]" if role else ""
                        line1 = font.render(f"{ch} {tag}".strip(), True, fg)
                        line2 = font.render(f"v={v:.2f}", True, fg)
                        line3 = font.render(f"{rgb[0]},{rgb[1]},{rgb[2]}", True, fg)
                        screen.blit(line1, (x + 4, y + 2))
                        screen.blit(line2, (x + 4, y + 18))
                        screen.blit(line3, (x + 4, y + 34))
            else:
                pad = 6
                for idx, (ch, rgb, v, role) in enumerate(items):
                    r = idx // max(1, self.cols)
                    c = idx % max(1, self.cols)
                    x0 = tile_x_off + c * self.scale + pad
                    y0 = r * self.scale + pad
                    w = self.scale - 2 * pad
                    h = self.scale - 2 * pad
                    pygame.draw.rect(screen, rgb, pygame.Rect(x0, y0, w, h))
                    pygame.draw.rect(screen, (30, 30, 34), pygame.Rect(x0, y0, w, h), 2)
                    if font:
                        fg = (10, 10, 10) if v > 0.65 else (240, 240, 240)
                        tag = f"[{role}]" if role else ""
                        line1 = font.render(f"{ch} {tag}".strip(), True, fg)
                        line2 = font.render(f"v={v:.2f}", True, fg)
                        line3 = font.render(f"{rgb[0]},{rgb[1]},{rgb[2]}", True, fg)
                        screen.blit(line1, (x0 + 4, y0 + 2))
                        screen.blit(line2, (x0 + 4, y0 + 18))
                        screen.blit(line3, (x0 + 4, y0 + 34))

            # Debug strip (between tiles and spectrum)
            if self.show_spectrum and self._dbg_h > 0:
                y_dbg = self._tiles_h
                pygame.draw.rect(screen, (14, 14, 18), pygame.Rect(0, y_dbg, self.w, self._dbg_h))
                pygame.draw.line(screen, (40, 40, 50), (0, y_dbg), (self.w, y_dbg), 1)

                if font:
                    # Small squares: Beat (B), Onset event (O), Kick (K)
                    box = 16
                    gap = 10
                    bx0 = 16
                    by0 = y_dbg + 10

                    def draw_box(ix: int, label: str, on: bool, color_on: Tuple[int, int, int]) -> None:
                        x = bx0 + ix * (box + gap)
                        pygame.draw.rect(screen, (40, 40, 50), pygame.Rect(x, by0, box, box), 1)
                        if on:
                            pygame.draw.rect(screen, color_on, pygame.Rect(x + 2, by0 + 2, box - 4, box - 4))
                        lab = font.render(label, True, (220, 220, 220))
                        screen.blit(lab, (x + 4, by0 + box + 4))

                    draw_box(0, "B", dbg_beat, (245, 245, 245))
                    draw_box(1, "O", dbg_onset_evt, (245, 245, 245))
                    draw_box(2, "K", dbg_kick, (220, 220, 70))
                    draw_box(3, "U", dbg_buildup, (255, 255, 255))  # bUildup
                    draw_box(4, "D", dbg_drop > 0.10, (255, 255, 255))  # Drop

                    # Onset meters (raw + env) with threshold marker
                    meter_w = 220
                    meter_h = 18
                    mx = bx0 + 5 * (box + gap) + 18
                    my = by0
                    pygame.draw.rect(screen, (40, 40, 50), pygame.Rect(mx, my, meter_w, meter_h), 1)
                    raw_n = clamp(dbg_onset / 3.0, 0.0, 1.0)
                    env_n = clamp(dbg_onset_env / 3.0, 0.0, 1.0)
                    pygame.draw.rect(screen, (230, 230, 230), pygame.Rect(mx + 1, my + 1, int((meter_w - 2) * raw_n), meter_h - 2))
                    pygame.draw.rect(screen, (80, 200, 255), pygame.Rect(mx + 1, my + 1, int((meter_w - 2) * env_n), max(2, int((meter_h - 2) * 0.45))))
                    thr_n = clamp(dbg_onset_thr / 3.0, 0.0, 1.0)
                    tx = mx + 1 + int((meter_w - 2) * thr_n)
                    pygame.draw.line(screen, (255, 90, 90), (tx, my + 1), (tx, my + meter_h - 2), 2)

                    # Text blocks (no overlap with spectrum now)
                    # Fixed-width formatting so the debug strip doesn't "jitter" as numbers change magnitude/sign.
                    line1 = f"onset[{dbg_onset_src or '?'}] o={dbg_onset:5.2f} env={dbg_onset_env:5.2f} thr={dbg_onset_thr:5.2f}  drop={dbg_drop:4.2f}"
                    line2 = (
                        f"tempo bpm={dbg_bpm:6.0f} conf={dbg_bpm_conf:4.2f} active={int(dbg_beat_active):1d} phase={dbg_beat_phase:4.2f}   "
                        f"dz={dbg_bass_dz:+6.2f} dbz={dbg_bass_db_z:+6.2f} dd={dbg_db_diff:+7.3f}   "
                        f"fd={dbg_flux_dev:+7.3f} fz={dbg_flux_z:+6.2f}"
                    )
                    t1 = font.render(line1, True, (210, 210, 210))
                    t2 = font.render(line2, True, (190, 190, 190))
                    screen.blit(t1, (mx, my + 24))
                    screen.blit(t2, (mx, my + 44))
                    if status_line:
                        t3 = font.render(status_line, True, (170, 170, 170))
                        screen.blit(t3, (mx, my + 64))

            # Spectrum panel (bottom)
            if self.show_spectrum:
                y0 = self._tiles_h + self._dbg_h
                pygame.draw.rect(screen, (8, 8, 10), pygame.Rect(0, y0, self.w, self.spectrum_h))
                # Plot area margins
                x_left = 60
                x_right = 10
                y_top = 14
                y_bottom = 48  # extra bottom space so x-axis label/ticks aren't clipped

                pygame.draw.line(screen, (40, 40, 50), (x_left, y0 + y_top), (x_left, y0 + self.spectrum_h - y_bottom), 1)
                pygame.draw.line(
                    screen,
                    (40, 40, 50),
                    (x_left, y0 + self.spectrum_h - y_bottom),
                    (self.w - x_right, y0 + self.spectrum_h - y_bottom),
                    1,
                )

                if freqs is not None and mag is not None and freqs.size == mag.size and freqs.size > 8:
                    # Log-frequency axis, full-range (20Hz..20kHz or Nyquist if lower)
                    f_min = 20.0
                    f_max = float(min(self.spectrum_max_hz, float(np.max(freqs))))
                    m = (freqs >= f_min) & (freqs <= f_max)
                    f = freqs[m]
                    y = mag[m]
                    if f.size > 8:
                        # dB scale (relative to current max) with fixed dynamic range for readable y-axis labels
                        y_db = 20.0 * np.log10(y + 1e-12)
                        y_db = y_db - float(np.max(y_db))  # <= 0
                        db_floor = -80.0
                        y_db = np.clip(y_db, db_floor, 0.0)
                        y_n = (y_db - db_floor) / (0.0 - db_floor)  # 0..1

                        # x = log10(f)
                        lf = np.log10(f)
                        lf0 = math.log10(f_min)
                        lf1 = math.log10(f_max)
                        xs = x_left + ((lf - lf0) / max(1e-9, (lf1 - lf0))) * (self.w - x_left - x_right)
                        ys = (y0 + self.spectrum_h - y_bottom) - y_n * (self.spectrum_h - y_top - y_bottom)
                        pts = [(int(xs[i]), int(ys[i])) for i in range(f.size)]
                        if len(pts) >= 2:
                            pygame.draw.lines(screen, (80, 220, 160), False, pts, 2)

                        # Band markers
                        def _x_for(freq_hz: float) -> int:
                            fx = float(max(f_min, min(freq_hz, f_max)))
                            lfx = math.log10(fx)
                            return int(x_left + ((lfx - lf0) / max(1e-9, (lf1 - lf0))) * (self.w - x_left - x_right))

                        # Low kick band (yellow)
                        for fx in (klo, khi):
                            x = _x_for(fx)
                            pygame.draw.line(screen, (220, 220, 70), (x, y0 + y_top), (x, y0 + self.spectrum_h - y_bottom), 2)
                        # High band (cyan)
                        for fx in (hlo, hhi):
                            x = _x_for(fx)
                            pygame.draw.line(screen, (80, 200, 255), (x, y0 + y_top), (x, y0 + self.spectrum_h - y_bottom), 2)
                        # Hat band (light gray)
                        for fx in (tlo, thi):
                            x = _x_for(fx)
                            pygame.draw.line(screen, (210, 210, 210), (x, y0 + y_top), (x, y0 + self.spectrum_h - y_bottom), 1)

                        if font:
                            # Labels
                            xlab = font.render("Frequency (Hz, log)", True, (200, 200, 200))
                            screen.blit(xlab, (x_left + 6, y0 + self.spectrum_h - 24))
                            ylab = font.render("Magnitude (dB rel)", True, (200, 200, 200))
                            ylab_r = pygame.transform.rotate(ylab, 90)
                            screen.blit(ylab_r, (2, y0 + y_top + 40))

                            # X ticks (log freq)
                            xticks = [20, 30, 40, 50, 60, 80, 100, 150, 200, 300, 400, 500, 1000, 2000, 5000, 10000, 20000]
                            xticks = [t for t in xticks if f_min <= t <= f_max]
                            for t in xticks:
                                ltx = math.log10(float(t))
                                xx = int(x_left + ((ltx - lf0) / max(1e-9, (lf1 - lf0))) * (self.w - x_left - x_right))
                                pygame.draw.line(screen, (40, 40, 50), (xx, y0 + self.spectrum_h - y_bottom), (xx, y0 + self.spectrum_h - y_bottom + 5), 1)
                                lab_txt = f"{int(t/1000)}k" if t >= 1000 else str(int(t))
                                lab = font.render(lab_txt, True, (180, 180, 180))
                                screen.blit(lab, (xx - 10, y0 + self.spectrum_h - y_bottom + 10))

                            # Y ticks (dB)
                            yticks = [0, -20, -40, -60, -80]
                            for tdb in yticks:
                                yn = (float(tdb) - db_floor) / (0.0 - db_floor)
                                yy = int((y0 + self.spectrum_h - y_bottom) - yn * (self.spectrum_h - y_top - y_bottom))
                                pygame.draw.line(screen, (40, 40, 50), (x_left - 5, yy), (x_left, yy), 1)
                                lab = font.render(str(int(tdb)), True, (180, 180, 180))
                                screen.blit(lab, (18, yy - 7))

                            txt = font.render(
                                f"spectrum (log-x, dB-y)   low kick: {klo:.0f}-{khi:.0f}Hz   high: {hlo:.0f}-{hhi:.0f}Hz",
                                True,
                                (230, 230, 230),
                            )
                            # Put the status text safely above the bottom edge (avoid clipping/"flipping")
                            screen.blit(txt, (x_left + 6, y0 + 2))

            pygame.display.flip()
            clock.tick(max(1.0, self.hz))

        pygame.quit()


# -----------------------------
# Main runtime
# -----------------------------

class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.stop_evt = threading.Event()
        self._last_ent_status: Optional[str] = None
        self._last_stream_proxy = None
        self._last_stats_print = 0.0
        self._live_lock = threading.Lock()
        self._live_snapshot: dict = {}

        # Channels
        if args.channels:
            self.channel_ids = [int(x) for x in args.channels]
        else:
            self.channel_ids = None
            if (not args.no_https_start) and (not args.no_auto_channels) and args.ent_id and args.app_key:
                try:
                    j = clip_v2_get_entertainment_configuration(args.bridge_ip, args.app_key, args.ent_id)
                    data0 = (j.get("data") or [{}])[0]
                    chs = [int(ch.get("channel_id")) for ch in (data0.get("channels") or []) if "channel_id" in ch]
                    chs = sorted(set(chs))
                    if chs:
                        self.channel_ids = chs
                except Exception:
                    self.channel_ids = None
            if not self.channel_ids:
                self.channel_ids = list(range(int(args.num_channels)))

        if len(self.channel_ids) < 1 or len(self.channel_ids) > 20:
            raise SystemExit("ERROR: number of channels must be 1..20")

        # Hue
        self.streamer = HueEntertainmentStreamer(
            bridge_ip=args.bridge_ip,
            clip_app_key=args.app_key,
            psk_identity=(args.psk_identity if args.psk_identity else args.app_key),
            clientkey_hex=args.clientkey_hex,
            entertainment_id=args.ent_id,
            channel_ids=self.channel_ids,
            dtls_backend=args.dtls_backend,
        )

        # Audio settings
        self.sr = int(args.sample_rate)
        self.update_hz = float(args.update_hz)
        self.dt = 1.0 / self.update_hz
        self._delay_lock = threading.Lock()
        self._delay_samples = max(0, int(round(float(args.audio_delay_ms) * self.sr / 1000.0)))

        # BPM estimator (onset-autocorrelation)
        self._bpm = 0.0
        self._bpm_conf = 0.0
        self._bpm_last_compute = 0.0
        # Multi-source onset buffers: different music emphasizes different bands.
        # We'll estimate BPM on multiple onset candidates and pick the one that is most periodic.
        self._onset_sources = ["mix", "kick", "high", "hat", "flux"]
        self._onset_buf_by: Dict[str, np.ndarray] = {
            k: np.zeros(int(self.update_hz * 16.0), dtype=np.float32) for k in self._onset_sources
        }
        self._onset_w = 0
        self._onset_filled = False
        self._onset_env_by: Dict[str, float] = {k: 0.0 for k in self._onset_sources}
        self._onset_src = "mix"
        # Per-source onset stats for event thresholding (phase lock + onset_rate)
        self._onset_mu_by: Dict[str, float] = {k: 0.0 for k in self._onset_sources}
        self._onset_var_by: Dict[str, float] = {k: 1e-6 for k in self._onset_sources}
        self._last_onset_evt_t_by: Dict[str, float] = {k: 0.0 for k in self._onset_sources}
        # Smoothed RMS gate for tempo logic (prevents silence from "finding a tempo")
        self._rms_env = 0.0

        # Beat grid (tempo-locked patterns)
        self._beat_period = 0.0
        self._next_beat_t = 0.0
        self._beat_idx = 0
        self._onset_events: List[float] = []
        self._double_burst_until = 0.0

        # Analyzer settings
        self.n_fft = int(args.fft)
        bands = (
            (20.0, 200.0),      # bass
            (200.0, 2000.0),    # mid
            (2000.0, 12000.0),  # treble
        )
        self.analyzer = FastAudioAnalyzer(
            sr=self.sr,
            n_fft=self.n_fft,
            bands=bands,
            perceptual_weighting=args.perceptual_weighting,
        )
        # Show-tuned bands (used for kick/high strength + spectrum markers)
        self.analyzer.set_kick_band(float(ShowOptions.KICK_BAND[0]), float(ShowOptions.KICK_BAND[1]))
        self.analyzer.set_high_band(float(ShowOptions.HIGH_BAND[0]), float(ShowOptions.HIGH_BAND[1]))
        if hasattr(self.analyzer, "set_hat_band") and hasattr(ShowOptions, "HAT_BAND"):
            self.analyzer.set_hat_band(float(ShowOptions.HAT_BAND[0]), float(ShowOptions.HAT_BAND[1]))
        self.show = EpicShow(self.channel_ids)

        # Optional preview window (single combined window: tiles + spectrum)
        self._preview: Optional[PreviewWindow] = None
        self._preview_thread: Optional[threading.Thread] = None
        if bool(args.preview):
            self._preview = PreviewWindow(
                channel_ids=self.channel_ids,
                scale=int(args.preview_scale),
                hz=float(args.preview_hz),
                layout=str(args.preview_layout),
                title="Hue Sync Debug",
                show_spectrum=True,
                spectrum_height=DEFAULT_SPECTRUM_HEIGHT,
                spectrum_max_hz=DEFAULT_SPECTRUM_MAX_HZ,
                window_width=0,
            )

        # Spectrum computation settings (separate FFT for higher-resolution display)
        self._last_spec_update = 0.0
        self._spec_fft_n = DEFAULT_SPECTRUM_FFT_N
        self._spec_win: Optional[np.ndarray] = None
        self._spec_win_n = 0

        # Ring buffer: keep ~2 seconds
        self.ring = AudioRing(max_samples=int(self.sr * 2.0))

        # For timing
        self._t0 = time.perf_counter()

    def _now(self) -> float:
        return time.perf_counter() - self._t0

    def request_stop(self) -> None:
        self.stop_evt.set()
        try:
            if self._preview:
                self._preview.stop()
        except Exception:
            pass

    def set_audio_delay_ms(self, ms: float) -> None:
        ms = max(0.0, float(ms))
        with self._delay_lock:
            self._delay_samples = max(0, int(round(ms * self.sr / 1000.0)))
        # keep args in sync for UI display / saving
        try:
            self.args.audio_delay_ms = ms
        except Exception:
            pass

    def get_audio_delay_ms(self) -> float:
        return float(self.args.audio_delay_ms)

    def _get_delay_samples(self) -> int:
        with self._delay_lock:
            return int(self._delay_samples)

    def _tune_delay_loop(self) -> None:
        """
        Interactive delay tuning (Linux TTY): press keys to adjust audio delay.
          [ / ] : -10ms / +10ms
          - / + : -50ms / +50ms
          0     : reset to 0ms
          s     : save to creds file
          q     : quit (stop program)
        """
        fd = sys.stdin.fileno()
        if not sys.stdin.isatty():
            print("NOTE: --tune-delay needs a TTY; skipping interactive tuning.", flush=True)
            return
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        try:
            print("Tune delay: [/-10ms  ]/+10ms  -/-50ms  +/+50ms  0=reset  s=save  q=quit", flush=True)
            while not self.stop_evt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not r:
                    continue
                ch = sys.stdin.read(1)
                cur = self.get_audio_delay_ms()
                if ch == "[":
                    self.set_audio_delay_ms(cur - 10.0)
                elif ch == "]":
                    self.set_audio_delay_ms(cur + 10.0)
                elif ch == "-":
                    self.set_audio_delay_ms(cur - 50.0)
                elif ch == "+" or ch == "=":
                    self.set_audio_delay_ms(cur + 50.0)
                elif ch == "0":
                    self.set_audio_delay_ms(0.0)
                elif ch.lower() == "s":
                    update_creds_file(self.args.creds_file, {"audio_delay_ms": float(self.get_audio_delay_ms())})
                    print(f"Saved audio_delay_ms={self.get_audio_delay_ms():.0f} to {self.args.creds_file}", flush=True)
                elif ch.lower() == "q":
                    self.request_stop()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    def _set_live_snapshot(self, **kw) -> None:
        with self._live_lock:
            self._live_snapshot.update(kw)

    def _get_live_snapshot(self) -> dict:
        with self._live_lock:
            return dict(self._live_snapshot)

    def _ui_loop(self) -> None:
        """
        Live terminal dashboard. Uses `rich` if available; falls back to ANSI.
        """
        hz = max(1.0, float(self.args.ui_hz))
        period = 1.0 / hz

        # Prefer rich if installed
        try:
            from rich.console import Console  # type: ignore
            from rich.table import Table  # type: ignore
            from rich.live import Live  # type: ignore

            console = Console()

            def render_table(s: dict) -> Table:
                t = Table(title="hue_sync live", show_header=False)
                t.add_column("k", style="bold cyan", no_wrap=True)
                t.add_column("v", style="white")
                for k, v in s.items():
                    t.add_row(str(k), str(v))
                return t

            with Live(render_table({}), console=console, refresh_per_second=hz, transient=True) as live:
                while not self.stop_evt.is_set():
                    live.update(render_table(self._get_live_snapshot()))
                    time.sleep(period)
            return
        except Exception:
            pass

        # ANSI fallback
        print("NOTE: `rich` not installed; using ANSI dashboard. Install with: pip install rich", flush=True)
        while not self.stop_evt.is_set():
            s = self._get_live_snapshot()
            lines = ["hue_sync live (ANSI)", "-" * 32]
            for k, v in s.items():
                lines.append(f"{k:>14}: {v}")
            block = "\n".join(lines)
            # Clear screen + home cursor
            print("\033[2J\033[H" + block, end="", flush=True)
            time.sleep(period)

    def start_entertainment_area(self) -> None:
        # Start entertainment area via HTTPS CLIP v2
        if self.args.no_https_start:
            print("NOTE: --no-https-start set; not starting entertainment area via CLIP v2.")
            return
        clip_v2_start_stop_entertainment(self.args.bridge_ip, self.args.app_key, self.args.ent_id, "start")
        # Best-effort: wait briefly for it to actually become active.
        try:
            t0 = time.time()
            while (time.time() - t0) < 2.0:
                j = clip_v2_get_entertainment_configuration(self.args.bridge_ip, self.args.app_key, self.args.ent_id)
                data0 = (j.get("data") or [{}])[0]
                self._last_ent_status = data0.get("status")
                self._last_stream_proxy = data0.get("stream_proxy")
                if self._last_ent_status == "active":
                    return
                time.sleep(0.15)
        except Exception:
            pass

    def stop_entertainment_area(self) -> None:
        if self.args.no_https_start:
            return
        try:
            clip_v2_start_stop_entertainment(self.args.bridge_ip, self.args.app_key, self.args.ent_id, "stop")
        except Exception:
            pass

    def run_file(self, path: str) -> None:
        # Open file
        f = sf.SoundFile(path, mode="r")
        if f.samplerate != self.sr:
            print(f"NOTE: file samplerate={f.samplerate}, resampling to {self.sr} for analysis/playback")
            # We'll use soundfile read + naive resample via numpy interpolation (fast enough for demos)
            # If you want higher quality, swap to resampy or soxr later.
            src_sr = int(f.samplerate)
            needs_resample = True
        else:
            src_sr = self.sr
            needs_resample = False

        channels = int(f.channels)

        # Audio callback reads frames, pushes mono into ring, outputs audio
        block = int(self.args.block)

        # Pre-read a little to avoid startup click
        f.read(0)

        def resample_linear(x: np.ndarray, src: int, dst: int) -> np.ndarray:
            if src == dst or x.size == 0:
                return x
            # x: (N,) or (N,C)
            n = x.shape[0]
            t_src = np.linspace(0.0, 1.0, n, endpoint=False)
            m = int(round(n * (dst / src)))
            t_dst = np.linspace(0.0, 1.0, m, endpoint=False)
            if x.ndim == 1:
                return np.interp(t_dst, t_src, x).astype(np.float32)
            out = []
            for c in range(x.shape[1]):
                out.append(np.interp(t_dst, t_src, x[:, c]).astype(np.float32))
            return np.stack(out, axis=1)

        def audio_cb(outdata, frames, tinfo, status):
            if status:
                # keep it quiet; status spam kills timing
                pass
            data = f.read(frames, dtype="float32", always_2d=True)
            if data.shape[0] == 0:
                raise sd.CallbackStop()

            if needs_resample:
                data_rs = resample_linear(data, src_sr, self.sr)
                # Keep output in original file rate if device expects sr? We opened stream at self.sr,
                # so we must output self.sr blocks. We'll just output resampled.
                data = data_rs
                if data.ndim == 1:
                    data = data[:, None]

            # Output
            if data.shape[0] < frames:
                # pad with zeros then stop
                pad = np.zeros((frames - data.shape[0], data.shape[1]), dtype=np.float32)
                data = np.vstack([data, pad])
                outdata[:] = data[:, :outdata.shape[1]]
                raise sd.CallbackStop()

            outdata[:] = data[:, :outdata.shape[1]]

            # Mono for analysis ring
            mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            self.ring.push(mono)

        # Start entertainment area first (opens UDP/2100 for DTLS on the bridge),
        # then connect the DTLS stream.
        try:
            print("Starting entertainment area…", flush=True)
            self.start_entertainment_area()
            if self.args.dtls_debug:
                try:
                    j = clip_v2_get_entertainment_configuration(self.args.bridge_ip, self.args.app_key, self.args.ent_id)
                    data0 = (j.get("data") or [{}])[0]
                    print(f"Entertainment status after start: {data0.get('status')!r}", flush=True)
                except Exception as ee:
                    print(f"NOTE: failed to read ent status: {type(ee).__name__}: {ee}", flush=True)

            print(f"Connecting DTLS ({self.args.dtls_backend})…", flush=True)
            self.streamer.connect()
        except Exception as e:
            # Snapshot status/proxy *before* we potentially stop it (helps debugging).
            ent_status = self._last_ent_status
            stream_proxy = self._last_stream_proxy
            if (not self.args.no_https_start):
                try:
                    j = clip_v2_get_entertainment_configuration(self.args.bridge_ip, self.args.app_key, self.args.ent_id)
                    data0 = (j.get("data") or [{}])[0]
                    ent_status = data0.get("status")
                    stream_proxy = data0.get("stream_proxy")
                except Exception:
                    pass

            # Ensure we don't leave the area running on failure (unless requested).
            if not self.args.keep_ent_active_on_fail:
                try:
                    self.stop_entertainment_area()
                finally:
                    self.streamer.close()

            msg = str(e)
            if type(e).__name__ == "TLSError" or "fatal alert" in msg.lower() or isinstance(e, TimeoutError) or "timed out" in msg.lower():
                # Pull extra fields if present (python-mbedtls TLSError exposes `.err` and `.msg`)
                tls_err = getattr(e, "err", None)
                tls_msg = getattr(e, "msg", None)
                tls_hex = hex(int(tls_err)) if isinstance(tls_err, int) else None

                extra = ""
                if self.args.dtls_debug:
                    extra_lines = []
                    extra_lines.append("")
                    extra_lines.append("DTLS debug:")
                    extra_lines.append(f"- TLSError.err: {tls_err!r}" + (f" ({tls_hex})" if tls_hex else ""))
                    extra_lines.append(f"- TLSError.msg: {tls_msg!r}")
                    extra_lines.append(f"- Exception repr: {repr(e)}")
                    extra_lines.append(f"- Exception args: {getattr(e, 'args', None)!r}")
                    extra_lines.append(f"- Bridge UDP target: {self.args.bridge_ip}:2100")
                    # Show what we think we're sending
                    extra_lines.append(f"- Offered cipher count: {len(getattr(self.streamer, '_offered_ciphers', []) or [])}")
                    if getattr(self.streamer, "_offered_ciphers", None):
                        extra_lines.append(f"- Offered ciphers (first 12): {self.streamer._offered_ciphers[:12]!r}")
                    extra_lines.append(f"- CLIP v2 ent status (snapshot): {ent_status!r}")
                    extra_lines.append(f"- stream_proxy (snapshot): {stream_proxy!r}")
                    extra_lines.append(f"- keep_ent_active_on_fail: {self.args.keep_ent_active_on_fail!r}")
                    extra = "\n".join(extra_lines) + "\n"

                raise SystemExit(
                    "\nERROR: DTLS handshake failed. This is usually a credential mismatch.\n"
                    f"- Bridge: {self.args.bridge_ip}\n"
                    f"- Entertainment area: {self.args.ent_id}\n"
                    f"- PSK identity (--psk-identity): {self.streamer.psk_identity!r}\n"
                    f"- PSK length (bytes): {len(self.streamer.psk)}\n\n"
                    + extra +
                    "Fix:\n"
                    "- Your `--app-key` is used for CLIP v2 HTTPS.\n"
                    "- For DTLS, hue_entertainment_pykit uses the bridge `hue-application-id` from `GET /auth/v1` as the PSK identity.\n"
                    "  Run `--auth` again (it stores `hue_application_id`) or pass it explicitly via `--psk-identity`.\n\n"
                    "How to create v1 username + clientkey (run after pressing the bridge link button):\n"
                    "  curl -s -X POST http://<BRIDGE_IP>/api \\\n"
                    "    -d '{\"devicetype\":\"hue_sync\",\"generateclientkey\":true}'\n\n"
                    "That returns a base64 `clientkey`. Convert it to hex:\n"
                    "  python3 - <<'PY'\n"
                    "import base64\n"
                    "print(base64.b64decode('<CLIENTKEY_BASE64>').hex())\n"
                    "PY\n"
                )
            raise

        # Analysis thread
        t_analysis = threading.Thread(target=self._analysis_loop, daemon=True)
        t_analysis.start()

        # Preview window thread (optional)
        if self._preview and (self._preview_thread is None):
            self._preview_thread = threading.Thread(
                target=self._preview.loop,
                kwargs={"on_quit": self.request_stop},
                daemon=True,
            )
            self._preview_thread.start()

        # UI thread (optional, re-renders in place)
        t_ui = None
        if self.args.ui:
            t_ui = threading.Thread(target=self._ui_loop, daemon=True)
            t_ui.start()

        # Tuning thread (optional)
        t_tune = None
        if self.args.tune_delay:
            t_tune = threading.Thread(target=self._tune_delay_loop, daemon=True)
            t_tune.start()

        # Play audio
        print(f"Playing: {path}")
        print("Press Ctrl+C to stop.")
        try:
            with sd.OutputStream(
                samplerate=self.sr,
                channels=min(2, channels),
                dtype="float32",
                blocksize=block,
                callback=audio_cb,
                device=self.args.device,
            ) as stream:
                # In file mode, analysis sees samples immediately in the callback, but you *hear* them
                # later due to output buffering. Auto-compensate preview/lights by adding the stream's
                # reported output latency to audio_delay_ms (unless user already set a big delay).
                try:
                    lat_s = float(getattr(stream, "latency", 0.0) or 0.0)
                    lat_ms = max(0.0, 1000.0 * lat_s)
                    # Only auto-apply when user hasn't intentionally dialed in a large delay already.
                    # (You can always override with --audio-delay-ms or --tune-delay.)
                    if self.get_audio_delay_ms() < 20.0 and lat_ms > 5.0:
                        self.set_audio_delay_ms(lat_ms)
                        print(f"Auto audio_delay_ms set from output latency: {lat_ms:0.0f}ms", flush=True)
                except Exception:
                    pass
                while not self.stop_evt.is_set():
                    time.sleep(0.05)
        finally:
            self.request_stop()
            t_analysis.join(timeout=1.0)
            if self._preview_thread:
                self._preview_thread.join(timeout=1.0)
            if t_ui:
                t_ui.join(timeout=1.0)
            if t_tune:
                t_tune.join(timeout=1.0)
            self._blackout()
            self.stop_entertainment_area()
            self.streamer.close()

    def run_dry_run(self, seconds: float) -> None:
        """
        Quick proof-of-life mode: start entertainment, DTLS handshake, stream a visible
        color cycle for `seconds`, then blackout and exit. No audio.
        """
        seconds = max(0.5, float(seconds))
        print("Starting entertainment area…", flush=True)
        self.start_entertainment_area()
        print(f"Connecting DTLS ({self.args.dtls_backend})…", flush=True)
        self.streamer.connect()

        print(f"Dry run: streaming for {seconds:.1f}s (Ctrl+C to stop)…", flush=True)
        t_end = time.time() + seconds
        try:
            while (time.time() < t_end) and (not self.stop_evt.is_set()):
                now_s = self._now()
                # simple rainbow cycle
                h = (now_s * 0.15) % 1.0
                rgb = {}
                for i, ch in enumerate(self.channel_ids):
                    hh = (h + (i / max(1, len(self.channel_ids))) * 0.25) % 1.0
                    r, g, b = hsv_to_rgb(hh, 1.0, 1.0)
                    rgb[ch] = rgb_to_u16(r, g, b, gamma=True)
                self.streamer.send_frame(rgb)
                self._set_live_snapshot(
                    t=f"{self._now():0.1f}s",
                    section="-",
                    mode="dry_run",
                    rms="-",
                    bass="-",
                    mid="-",
                    treble="-",
                    centroid="-",
                    flux="-",
                    riser="-",
                    kick="-",
                    channels=len(self.channel_ids),
                    ids=self.channel_ids,
                )
                time.sleep(1.0 / max(10.0, self.update_hz))
        finally:
            self._blackout()
            self.stop_entertainment_area()
            self.streamer.close()

    def run_system_audio(self) -> None:
        """
        Capture system audio (monitor/loopback input device) and drive the show in real-time.
        This does NOT output audio; it just listens and reacts.
        """
        # Start Hue first
        try:
            print("Starting entertainment area…", flush=True)
            self.start_entertainment_area()
            print(f"Connecting DTLS ({self.args.dtls_backend})…", flush=True)
            self.streamer.connect()
        except Exception:
            self._blackout()
            self.stop_entertainment_area()
            self.streamer.close()
            raise

        # Start analysis loop (lights)
        t_analysis = threading.Thread(target=self._analysis_loop, daemon=True)
        t_analysis.start()

        # Preview window thread (optional)
        if self._preview and (self._preview_thread is None):
            self._preview_thread = threading.Thread(
                target=self._preview.loop,
                kwargs={"on_quit": self.request_stop},
                daemon=True,
            )
            self._preview_thread.start()

        t_ui = None
        if self.args.ui:
            t_ui = threading.Thread(target=self._ui_loop, daemon=True)
            t_ui.start()

        t_tune = None
        if self.args.tune_delay:
            t_tune = threading.Thread(target=self._tune_delay_loop, daemon=True)
            t_tune.start()

        # Find an input device if not provided
        input_dev = self.args.input_device
        # argparse gives us strings; allow numeric indices like "9"
        if isinstance(input_dev, str):
            s = input_dev.strip()
            if s.isdigit():
                input_dev = int(s)
        if input_dev is None:
            try:
                devs = sd.query_devices()
                # Prefer monitor/loopback inputs
                for idx, d in enumerate(devs):
                    name = str(d.get("name", "")).lower()
                    if int(d.get("max_input_channels", 0)) > 0 and ("monitor" in name or "loopback" in name):
                        input_dev = idx
                        break
                # If no explicit monitor device exists (common with PipeWire/Pulse bridge),
                # prefer the generic "pulse" or "pipewire" device which maps to the default source
                # (and on your setup, the default source is the bluetooth .monitor).
                if input_dev is None:
                    for idx, d in enumerate(devs):
                        name = str(d.get("name", "")).lower()
                        if int(d.get("max_input_channels", 0)) > 0 and name in ("pulse", "pipewire", "default"):
                            input_dev = idx
                            break
            except Exception:
                input_dev = None

        if input_dev is None:
            print("ERROR: Could not auto-detect a system-audio monitor device.", flush=True)
            print("Run: python3 ./hue_sync.py --list-devices  (then pick one with --input-device)", flush=True)
            self.request_stop()
            t_analysis.join(timeout=1.0)
            self._blackout()
            self.stop_entertainment_area()
            self.streamer.close()
            return

        block = int(self.args.block)

        def in_cb(indata, frames, tinfo, status):
            if status:
                pass
            # indata shape: (frames, channels)
            mono = indata.mean(axis=1) if indata.ndim > 1 and indata.shape[1] > 1 else indata[:, 0]
            self.ring.push(mono)

        print(f"System audio mode: capturing from input device={input_dev!r}", flush=True)
        print("Press Ctrl+C to stop.", flush=True)
        try:
            with sd.InputStream(
                samplerate=self.sr,
                channels=2,
                dtype="float32",
                blocksize=block,
                callback=in_cb,
                device=input_dev,
            ):
                while not self.stop_evt.is_set():
                    time.sleep(0.05)
        finally:
            self.request_stop()
            t_analysis.join(timeout=1.0)
            if self._preview_thread:
                self._preview_thread.join(timeout=1.0)
            if t_ui:
                t_ui.join(timeout=1.0)
            if t_tune:
                t_tune.join(timeout=1.0)
            self._blackout()
            self.stop_entertainment_area()
            self.streamer.close()

    def _blackout(self) -> None:
        try:
            zeros = {ch: (0, 0, 0) for ch in self.channel_ids}
            self.streamer.send_frame(zeros)
        except Exception:
            pass

    def _analysis_loop(self) -> None:
        # Drive lights at update_hz
        next_t = time.perf_counter()
        last = time.perf_counter()
        window = int(self.args.window)

        while not self.stop_evt.is_set():
            now = time.perf_counter()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
                continue

            dt = now - last
            last = now
            next_t += self.dt

            # Grab last samples
            x = self.ring.read_last_offset(window, self._get_delay_samples())
            if x.size < 256:
                continue

            # Analyze
            f = self.analyzer.analyze(x, now_s=self._now())
            # Smoothed RMS for gating (more stable than instantaneous)
            self._rms_env = 0.95 * float(self._rms_env) + 0.05 * float(f.rms)

            # Multi-source onset envelopes for BPM estimation:
            # - "mix"  : analyzer's combined onset
            # - "kick" : kick strength
            # - "high" : high-band strength
            # - "hat"  : hat strength
            # - "flux" : positive spectral flux deviation (scaled)
            flux_dev = float(max(0.0, float(getattr(self.analyzer, "dbg_flux_dev", 0.0))))
            flux_gain = float(getattr(ShowOptions, "ONSET_FLUX_GAIN", 6.0))
            act = float(getattr(f, "activity", 0.0) or 0.0)
            # Use RAW strengths for BPM sources (visual strengths have a minimum scale, which is noisy in quiet sections).
            onset_raw_by = {
                "mix": float(getattr(f, "onset", 0.0)),
                "kick": float(getattr(f, "kick_strength_raw", 0.0)) * act,
                "high": float(getattr(f, "high_strength_raw", 0.0)) * act,
                "hat": float(getattr(f, "hat_strength_raw", 0.0)) * act,
                "flux": float(clamp(flux_gain * flux_dev, 0.0, 3.0)) * act,
            }
            # Smooth each envelope before storing (reduces jitter)
            # If audio is quiet, aggressively decay envelopes toward 0 so autocorr can't lock onto noise.
            stats_min_rms = float(getattr(ShowOptions, "BEAT_STATS_MIN_RMS", 0.0010))
            audio_ok = bool(self._rms_env >= stats_min_rms)
            for k, raw in onset_raw_by.items():
                prev = float(self._onset_env_by.get(k, 0.0))
                env = 0.90 * prev + 0.10 * (float(raw) if audio_ok else 0.0)
                self._onset_env_by[k] = env
                self._onset_buf_by[k][self._onset_w] = float(env)

            self._onset_w = (self._onset_w + 1) % next(iter(self._onset_buf_by.values())).size
            if self._onset_w == 0:
                self._onset_filled = True

            now_wall = time.perf_counter()
            if (now_wall - self._bpm_last_compute) > 0.5 and (self._onset_filled or self._onset_w > int(self.update_hz * 4.0)) and audio_ok:
                self._bpm_last_compute = now_wall
                # Pick the most periodic onset source
                best = (0.0, 0.0, self._onset_src)  # (bpm, conf, src)
                cur_src = str(self._onset_src)
                cur_best = (0.0, 0.0, cur_src)
                for src in self._onset_sources:
                    bpm_i, conf_i = self._estimate_bpm(src)
                    if src == cur_src:
                        cur_best = (bpm_i, conf_i, src)
                    if conf_i > best[1]:
                        best = (bpm_i, conf_i, src)

                bpm, conf, src = best
                # Hysteresis: don't switch sources unless clearly better (reduces flicker)
                if cur_best[1] > 0.0 and src != cur_src:
                    if conf < (cur_best[1] * 1.12 + 0.08):
                        bpm, conf, src = cur_best

                self._onset_src = str(src)
                if bpm > 0:
                    # Confidence gate + tempo lock:
                    # - If confidence is low, keep current bpm (avoid bouncing)
                    # - Allow jumps only with high confidence (real tempo change)
                    if conf < 1.6 and self._bpm > 0:
                        bpm = self._bpm
                    if self._bpm > 0 and abs(bpm - self._bpm) > max(10.0, 0.20 * self._bpm):
                        if conf < 2.4:
                            bpm = self._bpm

                    a = 0.18 if conf >= 2.4 else (0.10 if conf >= 2.0 else 0.04)
                    self._bpm = (1.0 - a) * self._bpm + a * bpm if self._bpm > 0 else bpm
                    self._bpm_conf = conf

            # Beat grid from bpm_est (gives "locked" patterns vs random reactivity)
            now_s = self._now()
            beat = False
            beat_phase = 0.0
            # Onset event threshold (also shown in pygame debug UI)
            # Use adaptive mean+std instead of env*constant (more robust across songs and volume levels).
            src = str(self._onset_src)
            o = float(self._onset_env_by.get(src, float(getattr(f, "onset", 0.0))))
            mu_a = 0.02  # slow stats, ~1s time constant at 50Hz
            beat_min_rms = float(getattr(ShowOptions, "BEAT_MIN_RMS", 0.0012))
            stats_min_rms = float(getattr(ShowOptions, "BEAT_STATS_MIN_RMS", 0.0010))
            # Prevent quiet sections from collapsing variance (which makes thresholds too low and causes false beats).
            if float(f.rms) >= stats_min_rms:
                mu0 = float(self._onset_mu_by.get(src, 0.0))
                var0 = float(self._onset_var_by.get(src, 1e-6))
                dev = o - mu0
                mu1 = (1.0 - mu_a) * mu0 + mu_a * o
                var1 = (1.0 - mu_a) * var0 + mu_a * (dev * dev)
                self._onset_mu_by[src] = float(mu1)
                self._onset_var_by[src] = float(var1)
            std = float(math.sqrt(max(1e-9, float(self._onset_var_by.get(src, 1e-6)))))
            k = float(getattr(ShowOptions, "ONSET_EVT_STD_K", 2.2))
            onset_thr = max(float(getattr(ShowOptions, "ONSET_EVT_FLOOR", 0.010)), float(self._onset_mu_by.get(src, 0.0)) + k * std)

            # Refractory prevents counting hats as multiple "events" in the same transient.
            ref = float(getattr(ShowOptions, "ONSET_EVT_REFRACTORY_S", 0.085))
            # Gate onset events in quiet sections to avoid hallucinated beat locks.
            last_evt = float(self._last_onset_evt_t_by.get(src, 0.0))
            is_onset_evt = (float(f.rms) >= beat_min_rms) and (o > onset_thr) and ((now_s - last_evt) >= ref)
            if is_onset_evt:
                self._last_onset_evt_t_by[src] = float(now_s)

            beat_conf_min = float(getattr(ShowOptions, "BEAT_CONF_MIN", 1.55))
            beat_min_rms_gate = float(getattr(ShowOptions, "BEAT_MIN_RMS", 0.0012))
            beat_active = bool(
                (self._rms_env >= beat_min_rms_gate)
                and (self._bpm > 50.0)
                and (self._bpm < 210.0)
                and (self._bpm_conf >= beat_conf_min)
            )
            if beat_active:
                period = 60.0 / max(1e-6, self._bpm)
                # initialize / retune gently
                if self._beat_period <= 0.0:
                    self._beat_period = period
                    self._next_beat_t = now_s + period
                else:
                    self._beat_period = 0.90 * self._beat_period + 0.10 * period
                    # keep next beat in a sane range
                    if (self._next_beat_t <= 0.0) or (self._next_beat_t < now_s - 2.0 * self._beat_period) or (self._next_beat_t > now_s + 2.0 * self._beat_period):
                        self._next_beat_t = now_s + self._beat_period

                # Phase lock using onset spikes near predicted beat
                if is_onset_evt:
                    self._onset_events.append(now_s)
                    # keep last ~2s
                    cutoff = now_s - 2.0
                    while self._onset_events and self._onset_events[0] < cutoff:
                        self._onset_events.pop(0)

                    err = (self._next_beat_t - now_s)
                    lock_win = min(0.10, 0.20 * self._beat_period)
                    if abs(err) < lock_win:
                        # pull beat grid toward the onset time (PLL-like)
                        target = now_s + self._beat_period
                        self._next_beat_t = 0.65 * self._next_beat_t + 0.35 * target

                    # Detect short double-time bursts (rapid onsets vs baseline beat)
                    if self._beat_period > 0.0 and len(self._onset_events) >= 4:
                        diffs = [self._onset_events[-i] - self._onset_events[-i - 1] for i in range(1, 4)]
                        diffs = [d for d in diffs if 0.04 < d < 0.60]
                        if diffs:
                            diffs.sort()
                            med = diffs[len(diffs) // 2]
                            if med < self._beat_period * 0.62 and self._bpm_conf >= 1.9:
                                self._double_burst_until = max(self._double_burst_until, now_s + 1.0)

                # Emit beat events when crossing beat times
                if self._next_beat_t > 0.0 and now_s >= self._next_beat_t:
                    # If we fell behind (CPU hiccup), catch up but emit only one beat per loop.
                    nskip = int((now_s - self._next_beat_t) / max(1e-6, self._beat_period))
                    self._beat_idx += max(1, nskip + 1)
                    self._next_beat_t = self._next_beat_t + (nskip + 1) * self._beat_period
                    beat = True

                if self._beat_period > 0.0 and self._next_beat_t > 0.0:
                    beat_phase = clamp(1.0 - (self._next_beat_t - now_s) / self._beat_period, 0.0, 1.0)

            # onset events per second (robust "drive" cue)
            onset_rate = 0.0
            if self._onset_events:
                cutoff = now_s - 1.0
                onset_rate = float(sum(1 for t in self._onset_events if t >= cutoff))

            f.beat = beat
            f.beat_phase = float(beat_phase)
            f.beat_idx = int(self._beat_idx)
            f.bpm_est = float(self._bpm)
            f.bpm_conf = float(self._bpm_conf)
            f.double_burst = bool(now_s < self._double_burst_until)
            f.onset_rate = float(onset_rate)

            # Update pygame spectrum + analyzer debug indicators (after beat/onset logic)
            if self._preview:
                try:
                    now_wall = time.perf_counter()
                    spec_hz = float(self.args.preview_hz)
                    if (now_wall - self._last_spec_update) >= (1.0 / max(1.0, spec_hz)):
                        self._last_spec_update = now_wall
                        n = int(self._spec_fft_n)
                        n = max(int(self.args.fft), min(n, 131072))
                        win = np.hanning(x.size).astype(np.float32)
                        xw = (x.astype(np.float32, copy=False) * win).astype(np.float32, copy=False)
                        spec = np.fft.rfft(xw, n=n)
                        mag = np.abs(spec).astype(np.float32) + 1e-12
                        freqs = np.fft.rfftfreq(n, d=1.0 / float(self.sr)).astype(np.float32)
                        self._preview.update_spectrum(
                            freqs,
                            mag,
                            float(ShowOptions.KICK_BAND[0]),
                            float(ShowOptions.KICK_BAND[1]),
                            float(self.analyzer.high_lo_hz),
                            float(self.analyzer.high_hi_hz),
                            float(getattr(self.analyzer, "hat_lo_hz", 8000.0)),
                            float(getattr(self.analyzer, "hat_hi_hz", 16000.0)),
                            float(o),
                            float(self._onset_env_by.get(str(self._onset_src), 0.0)),
                            float(onset_thr),
                            bool(is_onset_evt),
                            bool(f.beat),
                            bool(f.kick),
                            float(self._bpm),
                            float(self._bpm_conf),
                            bool(beat_active),
                            float(f.beat_phase),
                            str(self._onset_src),
                            bool(now_s < float(getattr(self.show, "_buildup_until", 0.0))),
                            float(getattr(self.show, "_drop_env", 0.0) or 0.0),
                            float(getattr(self.analyzer, "dbg_bass_dz", 0.0)),
                            float(getattr(self.analyzer, "dbg_bass_db_z", 0.0)),
                            float(getattr(self.analyzer, "dbg_db_diff", 0.0)),
                            float(getattr(self.analyzer, "dbg_flux_dev", 0.0)),
                            float(getattr(self.analyzer, "dbg_flux_z", 0.0)),
                        )
                except Exception:
                    pass

            # Map -> frame
            rgb = self.show.step(f, dt=dt, now_s=now_s)

            # Snapshot for UI/stats
            kick_bpm_val = float(getattr(self.show, "kick_bpm", 0.0) or 0.0)
            kick_bpm_str = f"{kick_bpm_val:0.0f}" if 40.0 <= kick_bpm_val <= 240.0 else "-"
            self._set_live_snapshot(
                t=f"{self._now():0.1f}s",
                section=getattr(self.show, "section", "beat"),
                mode=getattr(self.show, "mode", "?"),
                rms=f"{f.rms:0.3f}",
                bass=f"{f.bass:0.2f}",
                mid=f"{f.mid:0.2f}",
                treble=f"{f.treble:0.2f}",
                centroid=f"{f.centroid:0.0f}Hz",
                flux=f"{f.flux:0.3f}",
                riser=f"{f.riser:0.2f}",
                kick=int(f.kick),
                kick_strength=f"{getattr(f, 'kick_strength', 0.0):0.2f}",
                high=f"{getattr(f, 'high_strength', 0.0):0.2f}",
                palette=getattr(self.show, "_palettes", [("?", [], 0.0)])[getattr(self.show, "_palette_idx", 0)][0],
                beat=int(getattr(f, "beat", False)),
                beat_phase=f"{getattr(f, 'beat_phase', 0.0):0.2f}",
                burst=int(getattr(f, "double_burst", False)),
                onset_rate=f"{getattr(f, 'onset_rate', 0.0):0.1f}/s",
                delay_ms=f"{self.get_audio_delay_ms():0.0f}",
                last_kick_s=f"{(self._now() - getattr(self.show, 'last_kick_time', 0.0)):0.2f}",
                # Primary BPM: onset-autocorrelation estimator (robust)
                bpm=f"{self._bpm:0.0f}",
                bpm_conf=f"{self._bpm_conf:0.2f}",
                # Secondary/debug BPM: kick-based (can be wrong if kick detector chatters)
                kick_bpm=kick_bpm_str,
                bpm_est=f"{self._bpm:0.0f}",
                double_time=int(self._now() < getattr(self.show, '_double_time_until', 0.0)),
                # Kick debug (helps tuning)
                kick_hint="dz/db/flux based",
                channels=len(self.channel_ids),
                ids=self.channel_ids,
            )

            # Preview status line (helpful for matching what you see vs what you hear)
            if self._preview:
                try:
                    self._preview.set_status_line(f"audio_delay_ms={self.get_audio_delay_ms():0.0f}  (use --tune-delay for [ ] -/+10ms)")
                except Exception:
                    pass

            # Live stats HUD
            if self.args.stats:
                now_wall = time.perf_counter()
                hz = max(0.2, float(self.args.stats_hz))
                if (now_wall - self._last_stats_print) >= (1.0 / hz):
                    self._last_stats_print = now_wall
                    sec = getattr(self.show, "section", "?")
                    mode = getattr(self.show, "mode", "?")
                    print(
                        f"[{self._now():6.1f}s] section={sec:8s} mode={mode:5s} "
                        f"rms={f.rms:0.3f} bass={f.bass:0.2f} mid={f.mid:0.2f} treb={f.treble:0.2f} "
                        f"cent={f.centroid:5.0f}Hz flux={f.flux:0.3f} riser={f.riser:0.2f} kick={int(f.kick)} "
                        f"bpm={self._bpm:0.0f} conf={self._bpm_conf:0.2f} dt={int(self._now() < getattr(self.show, '_double_time_until', 0.0))} "
                        f"chs={len(self.channel_ids)} ids={self.channel_ids}",
                        flush=True,
                    )

            # Send
            try:
                self.streamer.send_frame(rgb)
                if self._preview:
                    self._preview.update_from_rgb16(rgb)
                    # Also show current group membership (kick/high) per channel in the preview labels.
                    try:
                        kick_sel = set(getattr(self.show, "_kick_sel", set()) or set())
                        high_sel = set(getattr(self.show, "_high_sel", set()) or set())
                        hat_sel = set(getattr(self.show, "_hat_sel", set()) or set())
                        self._preview.update_roles(kick_sel, high_sel, hat_sel)
                    except Exception:
                        pass
            except Exception:
                # If DTLS hiccups, don't hard-crash the show loop; you can restart.
                pass

    def _estimate_bpm(self, onset_src: str) -> Tuple[float, float]:
        """
        Estimate BPM from onset envelope via autocorrelation (tempogram-ish).
        Returns (bpm, confidence).
        """
        src = str(onset_src or "mix")
        if src not in self._onset_buf_by:
            src = "mix"
        # Reconstruct onset history in time order
        buf = self._onset_buf_by[src]
        if self._onset_filled:
            x = np.concatenate([buf[self._onset_w :], buf[: self._onset_w]])
        else:
            x = buf[: self._onset_w]
        if x.size < int(self.update_hz * 3.0):
            return 0.0, 0.0

        # Normalize
        x = x.astype(np.float32, copy=False)
        x = x - float(np.mean(x))
        x = x * np.hanning(x.size).astype(np.float32)

        # Autocorrelation via FFT (tempogram-ish)
        n = int(2 ** math.ceil(math.log2(max(16, x.size * 2))))
        X = np.fft.rfft(x, n=n)
        ac = np.fft.irfft(np.abs(X) ** 2, n=n)[: x.size]
        ac0 = float(ac[0]) + 1e-9
        ac = ac / ac0

        # BPM range
        bpm_min, bpm_max = 60.0, 190.0
        lag_min = int(self.update_hz * 60.0 / bpm_max)
        lag_max = int(self.update_hz * 60.0 / bpm_min)
        lag_min = max(2, min(lag_min, ac.size - 2))
        lag_max = max(lag_min + 1, min(lag_max, ac.size - 1))

        seg = ac[lag_min:lag_max]
        if seg.size < 3:
            return 0.0, 0.0

        # Harmonic reinforcement: tempo periodicity shows up at multiples.
        # Score each lag by combining 1x + 2x + 3x autocorr peaks.
        scores = np.zeros_like(seg, dtype=np.float32)
        for idx in range(seg.size):
            lag_i = lag_min + idx
            s = float(ac[lag_i])
            if (2 * lag_i) < ac.size:
                s += 0.55 * float(ac[2 * lag_i])
            if (3 * lag_i) < ac.size:
                s += 0.30 * float(ac[3 * lag_i])
            scores[idx] = s

        i = int(np.argmax(scores))
        lag = float(lag_min + i)
        peak = float(scores[i])

        # Quadratic interpolation around peak (on scores)
        if 1 <= i < scores.size - 1:
            y0, y1, y2 = float(scores[i - 1]), float(scores[i]), float(scores[i + 1])
            denom = (y0 - 2 * y1 + y2)
            if abs(denom) > 1e-9:
                delta = 0.5 * (y0 - y2) / denom
                lag = float(lag) + float(delta)

        bpm = float(60.0 * self.update_hz / float(lag))
        # Confidence: peak relative to average score in range
        conf = peak / (float(np.mean(scores)) + 1e-6)

        # Half/double-time handling: choose candidate closest to current bpm (if we have one)
        candidates = [(bpm, conf)]
        if bpm * 2 <= bpm_max:
            candidates.append((bpm * 2, conf * 0.95))
        if bpm / 2 >= bpm_min:
            candidates.append((bpm / 2, conf * 0.95))

        if self._bpm > 0:
            def score(c):
                b, cconf = c
                # prefer close to current bpm unless confidence is much higher
                dist = abs(math.log2(b / self._bpm))
                return cconf - 0.35 * dist
            bpm, conf = max(candidates, key=score)
        else:
            bpm, conf = max(candidates, key=lambda t: t[1])

        return float(bpm), float(conf)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EPIC Hue Entertainment audio sync (Linux)")

    p.add_argument("--bridge-ip", required=True, help="Hue bridge IP, e.g. 192.168.1.2")
    p.add_argument("--creds-file", default=DEFAULT_CREDS_FILE,
                   help="Path to store/load Hue credentials JSON (default: ./hue_creds.json)")
    p.add_argument("--auth", action="store_true",
                   help="Authenticate with the bridge (press link button) and store creds to --creds-file, then exit.")
    p.add_argument("--auth-timeout", type=float, default=30.0,
                   help="Seconds to wait for link button when using --auth (default: 30)")

    p.add_argument("--app-key", default=None,
                   help="Hue application key for CLIP v2 HTTPS (hue-application-key header). "
                        "If omitted, loaded from --creds-file. "
                        "Tip: you can pass this during --auth to keep your existing CLIP v2 key while generating DTLS creds.")
    p.add_argument("--psk-identity", default=None,
                   help="DTLS PSK identity for Entertainment streaming. "
                        "If omitted, loaded from --creds-file. "
                        "hue_entertainment_pykit uses the bridge `hue-application-id` from GET /auth/v1 as the DTLS identity. "
                        "If not set and not in creds, defaults to --app-key.")
    p.add_argument("--clientkey-hex", default=None,
                   help="Hue clientkey in HEX (PSK). If omitted, loaded from --creds-file.")
    p.add_argument("--ent-id", default=None, help="Entertainment configuration UUID (36 chars)")

    p.add_argument("--file", default=None, help="Audio file path to play (wav/mp3/flac/etc)")
    p.add_argument("--dry-run", action="store_true",
                   help="Do not play audio. Just connect DTLS and stream a short color cycle, then exit.")
    p.add_argument("--dry-run-seconds", type=float, default=5.0,
                   help="Seconds to stream during --dry-run (default: 5)")

    # System audio capture (desktop output / "monitor" source)
    p.add_argument("--system-audio", action="store_true",
                   help="Analyze system audio output (PulseAudio/PipeWire monitor) instead of playing a file.")
    p.add_argument("--input-device", default=None,
                   help="sounddevice input device (name or index). For system audio, pick a 'monitor' device. "
                        "Use --list-devices to see options.")
    p.add_argument("--list-devices", action="store_true",
                   help="List sounddevice input/output devices and exit.")

    p.add_argument("--sample-rate", type=int, default=48000, help="Playback + analysis sample rate")
    p.add_argument("--device", default=None, help="sounddevice output device (name or index)")
    p.add_argument("--block", type=int, default=1024, help="Audio callback blocksize")
    p.add_argument("--window", type=int, default=4096, help="Analysis window (samples)")
    p.add_argument("--audio-delay-ms", type=float, default=0.0,
                   help="Delay lights relative to captured audio (ms). Useful for Bluetooth latency. Default: 0.")
    p.add_argument("--tune-delay", action="store_true",
                   help="Interactive delay tuning keys while running ([ ] -/+10ms, -/+50ms, s=save, q=quit).")

    p.add_argument("--update-hz", type=float, default=50.0, help="Hue stream update rate (recommended 50–60Hz)")

    p.add_argument("--fft", type=int, default=4096, help="FFT size (should be >= window)")
    p.add_argument(
        "--perceptual-weighting",
        choices=["none", "a"],
        default="a",
        help="Perceptual frequency weighting for spectral features (bands/flux/BPM). 'a' better matches human hearing; 'none' is raw.",
    )
    p.add_argument("--preview", action="store_true", help="Show a local pygame debug window (per-channel tiles + spectrum analyzer).")
    p.add_argument("--preview-hz", type=float, default=DEFAULT_PREVIEW_HZ, help="Preview window refresh rate (Hz). Also used for spectrum refresh.")
    p.add_argument("--preview-scale", type=int, default=DEFAULT_PREVIEW_SCALE, help="Preview tile size in pixels.")
    p.add_argument("--preview-layout", choices=["grid", "circle"], default=DEFAULT_PREVIEW_LAYOUT, help="Preview layout.")
    p.add_argument("--num-channels", type=int, default=6, help="How many Hue Entertainment channels to drive (if --channels not set)")
    p.add_argument("--channels", type=int, nargs="*", default=None, help="Explicit channel IDs (e.g. 0 1 2 3 4 5)")

    ch_group = p.add_mutually_exclusive_group()
    ch_group.add_argument("--auto-channels", action="store_true", default=True,
                          help="Auto-detect channel IDs from the entertainment configuration (default).")
    ch_group.add_argument("--no-auto-channels", action="store_true",
                          help="Disable auto channel detection; use --num-channels (or --channels).")

    stats_group = p.add_mutually_exclusive_group()
    stats_group.add_argument("--stats", action="store_true", default=True,
                             help="Print live show stats (section/mode/energy/bands/etc) while running (default).")
    stats_group.add_argument("--no-stats", action="store_false", dest="stats",
                             help="Disable live show stats printing.")
    p.add_argument("--stats-hz", type=float, default=2.0,
                   help="Stats print rate in Hz when --stats is set (default: 2.0)")

    ui_group = p.add_mutually_exclusive_group()
    ui_group.add_argument("--ui", action="store_true", default=False,
                          help="Use a live terminal dashboard (re-renders in place). Prefer `rich` if installed.")
    ui_group.add_argument("--no-ui", action="store_false", dest="ui",
                          help="Disable the live terminal dashboard.")
    p.add_argument("--ui-hz", type=float, default=12.0,
                   help="Dashboard refresh rate in Hz when --ui is set (default: 12)")

    p.add_argument("--no-https-start", action="store_true",
                   help="Do NOT start/stop entertainment area via CLIP v2 HTTPS. "
                        "Only use if you start the area manually some other way.")

    p.add_argument("--dtls-debug", action="store_true",
                   help="Print extra DTLS debug info on handshake failures (cipher list, TLSError codes, ent status).")
    p.add_argument("--keep-ent-active-on-fail", action="store_true",
                   help="If DTLS handshake fails, do not auto-stop the entertainment area (helps debugging).")

    p.add_argument("--dtls-backend", choices=["pykit", "mbedtls"], default="pykit",
                   help="DTLS implementation to use. 'pykit' uses hue_entertainment_pykit/network.dtls (recommended). "
                        "'mbedtls' uses the local python-mbedtls codepath.")

    return p.parse_args()


def main() -> None:
    args = parse_args()

    # If using a dashboard, disable line-by-line stats unless user explicitly forced them.
    if args.ui:
        args.stats = False

    # List audio devices and exit (utility)
    if args.list_devices:
        try:
            print(sd.query_devices())
        except Exception as e:
            raise SystemExit(f"ERROR: failed to query devices: {e}")
        return

    # --auth: register with the bridge and store creds, then exit
    if args.auth:
        # Load existing creds (if any) so we can preserve an already-working CLIP v2 key.
        existing = {}
        if args.creds_file and os.path.exists(args.creds_file):
            try:
                existing0 = load_creds(args.creds_file)
                existing, _ = maybe_migrate_creds(existing0)
            except Exception:
                existing = {}

        print(f"Auth mode: press the Hue Bridge link button, waiting up to {args.auth_timeout:.0f}s…")
        reg = hue_v1_register_with_button(args.bridge_ip, timeout_s=float(args.auth_timeout))

        # CLIP v2 app key is used only for HTTPS start/stop. Prefer:
        # 1) explicitly provided --app-key
        # 2) existing creds file app key
        # 3) fall back to the newly-created username (often works, but not always)
        clip_app_key = (
            args.app_key
            or existing.get("clip_app_key")
            or existing.get("app_key")
            or reg["username"]
        )

        # hue_entertainment_pykit uses the bridge `hue-application-id` header as DTLS identity.
        hue_app_id = clip_v2_get_hue_application_id(args.bridge_ip, clip_app_key)

        creds = {
            "bridge_ip": args.bridge_ip,
            # HTTPS CLIP v2 key (header: hue-application-key)
            "clip_app_key": clip_app_key,
            # Backwards-compat: keep app_key too
            "app_key": clip_app_key,
            # v1 registration output (useful reference)
            "username": reg["username"],
            # DTLS identity + key for Entertainment streaming (pykit-style)
            "hue_application_id": hue_app_id,
            "psk_identity": hue_app_id,
            "clientkey_hex": reg["clientkey_hex"],
            "clientkey_b64": reg["clientkey_b64"],
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        save_creds(args.creds_file, creds)
        print(f"Saved credentials to: {args.creds_file}")
        print("Next run: omit --app-key/--clientkey-hex/--psk-identity and they will be loaded from the creds file.")
        return

    # Load creds file if CLI omitted values
    if (args.app_key is None) or (args.clientkey_hex is None) or (args.psk_identity is None):
        if args.creds_file and os.path.exists(args.creds_file):
            c0 = load_creds(args.creds_file)
            c, changed = maybe_migrate_creds(c0)
            if changed:
                # Keep the on-disk file consistent so future runs are correct.
                save_creds(args.creds_file, c)
                print(f"NOTE: migrated creds file in place: {args.creds_file}")
            if args.app_key is None:
                args.app_key = c.get("clip_app_key") or c.get("app_key")
            if args.clientkey_hex is None:
                args.clientkey_hex = c.get("clientkey_hex")
            if args.psk_identity is None:
                args.psk_identity = (
                    c.get("hue_application_id")
                    or c.get("psk_identity")
                    or c.get("dtls_identity")
                    or c.get("username")
                )
            if args.audio_delay_ms == 0.0 and isinstance(c.get("audio_delay_ms"), (int, float, str)):
                try:
                    args.audio_delay_ms = float(c.get("audio_delay_ms"))
                except Exception:
                    pass

    if not args.app_key:
        raise SystemExit("ERROR: Missing --app-key (or run with --auth to create a creds file).")
    if not args.clientkey_hex:
        raise SystemExit("ERROR: Missing --clientkey-hex (or run with --auth to create a creds file).")
    if not args.ent_id:
        raise SystemExit("ERROR: Missing --ent-id")
    if (not args.dry_run) and (not args.system_audio) and (not args.file):
        raise SystemExit("ERROR: Missing --file (or use --system-audio / --dry-run)")

    # Basic sanity: FFT should be >= window
    if args.fft < args.window:
        print("NOTE: --fft < --window; forcing fft=window")
        args.fft = args.window

    runner = Runner(args)

    def on_sigint(sig, frame):
        runner.request_stop()
    signal.signal(signal.SIGINT, on_sigint)
    signal.signal(signal.SIGTERM, on_sigint)

    if args.dry_run:
        runner.run_dry_run(float(args.dry_run_seconds))
    elif args.system_audio:
        runner.run_system_audio()
    else:
        runner.run_file(args.file)


if __name__ == "__main__":
    main()
