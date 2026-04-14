---
name: id-ocr
slug: id-ocr
version: 1.0.4
description: Extract a person’s identity fields from an uploaded ID document image, including transliterated English name and normalized dates.
---

## When to Use

Use this skill when the user types **"id-ocr"** (or asks to OCR an ID document) and wants structured identity details extracted from an ID image.

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

## Optional: ID Photo Extraction (OpenCV)

If you need to extract the **ID photo rectangle** (not just a headshot crop), you can use the bundled helper scripts under `scripts/`.

### Requirements (WSL/Ubuntu)

```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy
```

### Model weights

Download OpenCV DNN face detector weights into:

`models/opencv-face/`

```bash
curl -L -o models/opencv-face/deploy.prototxt \
  https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt

curl -L -o models/opencv-face/res10_300x300_ssd_iter_140000_fp16.caffemodel \
  https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel
```

### Scripts

- `scripts/crop_headshot_opencv_dnn.py` — detect face and produce a padded headshot crop (hair-to-chin style).
- `scripts/crop_id_photo_opencv.py` — attempt to crop the **ID photo rectangle**; falls back to a conservative face-based rectangle if border detection fails.
