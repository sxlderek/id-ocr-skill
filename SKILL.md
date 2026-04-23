---
name: id-ocr
slug: id-ocr
version: 1.1.0
description: Extract a person’s identity fields from an uploaded ID document image, including transliterated English name and normalized dates.
---

## When to Use

Use this skill when the user types **"id-ocr"** (or asks to OCR an ID document) and wants structured identity details extracted from an ID image.

## Core Rules

1) **Require an ID image first**
   - If the user has not provided an ID document image in the current conversation, ask them to upload a clear photo/scan of the ID (front, and back if relevant).
   - Assume the document can be in **any language**. Ask the user which language/country it is if unsure.

2) **Output exactly these fields** (in this order)
   1. Full Name (Native)
   2. Full Name (EN) — First name + Last name in English (best-effort phonetic transliteration; mixed case)
   3. DOB (YYYY-MM-DD)
   4. Document type (one of: National ID, Driving License, Passport, HK/Macau Pass(港澳通行证), or `[unclear]`)
   5. Document number (ID/passport/permit number; or `[unclear]`)
   6. Issuing country of the document
   7. Expiry (YYYY-MM-DD | N/A | [unclear])

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

Return the results as 7 lines, exactly like:

Full Name (Native): ...
Full Name (EN): First Last
DOB: YYYY-MM-DD
Document type: National ID | Driving License | Passport | HK/Macau Pass(港澳通行证) | [unclear]
Document number: ... | [unclear]
Issuing country: ...
Expiry: YYYY-MM-DD | N/A | [unclear]

## Optional: Extract ID Photo + Generate Artistic Avatar

If the user asks for an **artistic avatar** (e.g., “make a social avatar”), do this after the 7-line output:

1) **Preprocess if needed (rotate + flatten)**
   - If the ID is tilted, run a best-effort card-boundary detection + perspective transform to flatten it before cropping.

2) **Extract the ID photo rectangle** from the document and send it back to the user.
   - Add padding and **verify the crop contains the full face (hairline → neck)** before sending.
   - Verification method (preferred): use **MediaPipe FaceMesh** to confirm mouth + chin + forehead landmarks are present with margin (not cut off). If it fails, retry with progressively larger padding and/or re-crop.
   - If you cannot reliably verify (repeated failures), ask the user for a clearer photo of the ID (less glare, higher resolution, straighter angle).
2) Ask the user for:
   - desired style (e.g., anime, watercolor, oil painting, Pixar-like, cyberpunk)
   - background (solid color / gradient / transparent)
   - aspect ratio (1:1 recommended for socials)
3) **Get explicit confirmation** before generating the avatar if the workflow requires sending the cropped photo to an external model/provider.
4) Generate 1–3 avatar variants and send them.

See `avatar.md` for prompt presets.

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
- `scripts/flatten_id_card.py` — rotate/flatten (deskew) the ID card via card-boundary detection.
- `scripts/mediapipe_facecheck.py` — verify crop contains full face (forehead/mouth/chin) using MediaPipe FaceMesh.
- `scripts/extract_id_photo_fullface.py` — flatten (best-effort) + photo-rect crop + **FaceMesh verification** with aggressive padding retries (recommended).
- `scripts/extract_id_photo_rect_verified.py` — edge-based photo-rectangle crop + basic verification + padding retries (baseline).
