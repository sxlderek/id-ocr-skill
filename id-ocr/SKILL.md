---
name: id-ocr
slug: id-ocr
version: 1.0.3
description: Extract a person’s identity fields from an uploaded ID document image, including transliterated English name and normalized dates.
---

## When to Use

Use this skill when the user types **"IDOCR"** (or asks to OCR an ID document) and wants structured identity details extracted from an ID image.

## Core Rules

1) **Require an ID image first**
   - If the user has not provided an ID document image in the current conversation, ask them to upload a clear photo/scan of the ID (front, and back if relevant).
   - Assume the document can be in **any language**. Ask the user which language/country it is if unsure.

2) **Output exactly these fields** (in this order)
   - Full name in local language
   - First name + Last name in English (best-effort phonetic transliteration)
   - Date of birth (YYYY-MM-DD)
   - Document expiry date (if any) (YYYY-MM-DD)
   - Issuing country of the document

3) **Be explicit about uncertainty**
   - If any field is missing/illegible, output it as: `[unclear]`.
   - If the document has **no expiry date**, output: `N/A`.

4) **English name casing (must be mixed case)**
   - Always output `Name (EN)` in **mixed case** (e.g., `Chunyue Liang`), not ALL CAPS.
   - If the ID prints the English name in ALL CAPS, convert it to mixed case.

5) **Transliteration rules (phonetic translate)**
   - If the ID does not provide an English/Latin name, produce a best-effort phonetic transliteration.
   - Prefer a language-appropriate romanization when obvious (examples: Chinese → pinyin; Japanese → Hepburn; Korean → RR; Thai → RTGS; Arabic → common passport-style Latinization).
   - If the script/language is unclear, ask a follow-up question (or provide a best-effort transliteration and mark `[unclear]`).

6) **Date normalization**
   - Convert any detected date format (e.g., DD/MM/YYYY, MM-DD-YY, local formats) into **YYYY-MM-DD**.
   - If only partial date is present (month/year only), output `[unclear]`.

7) **Privacy + minimal retention**
   - Do not store or repeat unnecessary sensitive details (ID number, address, MRZ, etc.) unless the user explicitly asks.

## Response Format

Return the results as 5 lines, exactly like:

Full name (local): ...
Name (EN): First Last
DOB: YYYY-MM-DD
Expiry: YYYY-MM-DD | N/A | [unclear]
Issuing country: ...
