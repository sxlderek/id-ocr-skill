# idocr-skill

OpenClaw skill: **IDOCR** — Extract a person’s identity fields from an uploaded ID document image, including transliterated English name and normalized dates.

## What it returns (exact fields)

- Full name (local)
- Name (EN): First Last (mixed case)
- DOB: YYYY-MM-DD
- Expiry: YYYY-MM-DD | N/A | [unclear]
- Issuing country

## Privacy

The skill is designed to avoid emitting unnecessary sensitive data (ID number, address, MRZ, etc.) unless explicitly requested.

## Files

- `IDOCR/SKILL.md` — skill instructions
