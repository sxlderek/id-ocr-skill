# id-ocr-skill

OpenClaw skill: **id-ocr** — Extract key identity fields from an uploaded ID document image and return normalized, structured output.

## Trigger

Type `id-ocr` in chat, then upload a clear photo/scan of the ID (front, and back if relevant).

## Output (exact fields)

- Full name (local)
- Name (EN): First Last (mixed case; best-effort phonetic transliteration)
- DOB: YYYY-MM-DD
- Expiry: YYYY-MM-DD | N/A | [unclear]
- Issuing country

## Privacy

The skill avoids emitting unnecessary sensitive data (ID number, address, MRZ, etc.) unless explicitly requested.

## Files

- `SKILL.md` — skill instructions
- `extraction-notes.md` — extraction notes (on-demand)
