# id-ocr-skill (OpenClaw Skill)

Extract key identity fields from an uploaded ID document image and return normalized, structured output.

Also includes **offline ID photo extraction** with FaceMesh verification and optional avatar generation flow.

## Trigger

Type `id-ocr` in chat, then upload an ID image (front, and back if relevant).

## Output (7 lines, in order)

1) Full Name (Native)
2) Full Name (EN)
3) DOB (YYYY-MM-DD)
4) Document type
5) Document number
6) Issuing country
7) Expiry (YYYY-MM-DD / N/A / [unclear])

## Offline ID photo extraction

Key scripts:

- `scripts/flatten_id_card.py` — best-effort rotate/flatten (deskew)
- `scripts/extract_id_photo_fullface.py` — extract portrait with **MediaPipe FaceMesh** verification (forehead/mouth/chin)
- `scripts/mediapipe_facecheck.py` — standalone FaceMesh verification helper

## Privacy

The skill avoids emitting unnecessary sensitive details unless explicitly requested.
