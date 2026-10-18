#!/usr/bin/env bash
# Download the external Kaggle datasets catalogued in data/kaggle/DATASETS.md
# into data/kaggle/<owner>__<slug>/ (~2.7 GB). Already-present datasets are skipped.
#
# Requires the Kaggle CLI (`pip install kaggle`) and an API token in
# ~/.kaggle/kaggle.json (Kaggle → Settings → API → Create New Token).
set -euo pipefail

DEST="$(cd "$(dirname "$0")/.." && pwd)/data/kaggle"

DATASETS=(
  # A. Clinical text with ICD-10 labels
  devp1866/noisy-medical-document-images-ocr
  serdarcaglar/medprocner-codiesp-spanish-medical-texts
  # B. Tabular ICD-10
  generaterealdata/synthetic-hospital-records
  sergionefedov/patient-records-100k-patients-15-conditions
  nudratabbas/healthcare-fraud-detection-dataset
  # Reference code sets / vocabularies
  mrhell/icd10cm-codeset-2023
  arashnic/icd10-codes-and-descriptions
  ahsanneural/rare-diseases-orphadata-2026
  nguyenchuhunganh/medical-code
  # C. Medical text without ICD
  imeritinc/multilingual-ambient-scribe-dataset
  tboyle10/medicaltranscriptions
  priyamchoksi/pmc-patients-dataset-for-clinical-decision-support
  cnjameshu/nbme-patient-notes
  sadiqguru/medical-record-sample-data
  thedevastator/medical-conversation-corpus-100k
  niyarrbarman/symptom2disease
  artemminiailo/medicalconversations2disease
  # D. PII detection
  alejopaullier/pii-external-dataset
  thedrcat/persuade-pii-dataset
  newtonbaba12345/pii-detection-gemini-created-dataset
  pjmathematician/pii-detection-dataset-gpt
  krist0phersmith/pii-real-names-dataset
)

command -v kaggle >/dev/null || { echo "kaggle CLI not found: pip install kaggle" >&2; exit 1; }

failed=()
for ref in "${DATASETS[@]}"; do
  dir="$DEST/${ref/\//__}"
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir")" ]; then
    echo "skip     $ref (already present)"
    continue
  fi
  mkdir -p "$dir"
  echo "download $ref"
  if ! kaggle datasets download "$ref" -p "$dir" --unzip -q; then
    echo "FAILED   $ref" >&2
    rm -rf "$dir"
    failed+=("$ref")
  fi
done

# The names dataset bundles an untrusted Windows binary that we don't need.
rm -f "$DEST/krist0phersmith__pii-real-names-dataset/0717-182/0717-182/gender.exe"

if [ ${#failed[@]} -gt 0 ]; then
  echo "Failed: ${failed[*]}" >&2
  exit 1
fi
echo "Done: ${#DATASETS[@]} datasets in $DEST"
