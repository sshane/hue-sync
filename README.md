## hue_sync

Streams Philips Hue **Entertainment** frames driven by audio analysis (system-audio or file playback).

### Security / publishing note
Do **NOT** commit Hue credentials to git.
- Local credentials live in `hue_creds.json` (ignored by `.gitignore`)
- If you use `pykit_auth_test.py`, it may write `data/auth.json` (also ignored)

If you accidentally committed secrets in the past, rotate them in the Hue app / re-auth and rewrite history.

### Install

```bash
pip install numpy sounddevice soundfile requests python-mbedtls pygame
```

### Authenticate (creates local `hue_creds.json`)
Press the physical Hue bridge button, then run:

```bash
python3 ./hue_sync.py --bridge-ip <BRIDGE_IP> --auth
```

### Run (system audio)

```bash
python3 ./hue_sync.py \
  --bridge-ip <BRIDGE_IP> \
  --ent-id <ENTERTAINMENT_CONFIG_UUID> \
  --system-audio \
  --input-device <DEVICE_INDEX_OR_NAME> \
  --preview
```

Tips:
- Use `python3 ./hue_sync.py --list-devices` to find a monitor/loopback input.
- If the preview/lights look early/late vs what you hear, use `--tune-delay` or `--audio-delay-ms`.
