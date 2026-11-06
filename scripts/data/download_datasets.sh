#!/usr/bin/env bash
# Download the original public datasets without curating or fitting anything.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

LLMAIL_DIR="data/llmail"
LLMAIL_BASE="https://huggingface.co/datasets/microsoft/llmail-inject-challenge/resolve/main/data"

mkdir -p "$LLMAIL_DIR"

download() {
  local url="$1"
  local output="$2"
  local expected_bytes="$3"
  local partial="${output}.part"

  # Reuse a complete file; resume incomplete downloads in a separate .part file.
  if [[ -f "$output" ]] && [[ "$(stat -c %s "$output")" == "$expected_bytes" ]]; then
    echo "verified $output"
    return
  fi

  echo "downloading $output"
  curl --location --fail --retry 5 --continue-at - --output "$partial" "$url"
  if [[ "$(stat -c %s "$partial")" != "$expected_bytes" ]]; then
    echo "size check failed for $output" >&2
    exit 1
  fi
  # Rename only after the size matches the recorded release. This is not a hash check.
  mv "$partial" "$output"
}

# Download only the LLMail sources used to prepare the retained testing data.
download "$LLMAIL_BASE/emails_for_fp_tests.json" "$LLMAIL_DIR/emails_for_fp_tests.json" 55486
download "$LLMAIL_BASE/labelled_unique_submissions_phase2.json" "$LLMAIL_DIR/labelled_unique_submissions_phase2.json" 68586935
download "$LLMAIL_BASE/levels_descriptions.json" "$LLMAIL_DIR/levels_descriptions.json" 884
download "$LLMAIL_BASE/objectives_descriptions.json" "$LLMAIL_DIR/objectives_descriptions.json" 379
download "$LLMAIL_BASE/scenarios.json" "$LLMAIL_DIR/scenarios.json" 38812
download "$LLMAIL_BASE/system_prompt.json" "$LLMAIL_DIR/system_prompt.json" 3065

echo "dataset download complete"
