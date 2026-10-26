#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-raw_data/external}"
SDD_ROOT="${ROOT}/sdd_annotations_subset"
mkdir -p "${SDD_ROOT}/annotations/bookstore/video0"
mkdir -p "${SDD_ROOT}/annotations/bookstore/video1"
mkdir -p "${SDD_ROOT}/annotations/gates/video0"
mkdir -p "${SDD_ROOT}/annotations/gates/video1"
mkdir -p "${SDD_ROOT}/annotations/nexus/video0"
mkdir -p "${SDD_ROOT}/annotations/nexus/video1"

download_blob() {
  local repo="$1"
  local sha="$2"
  local target="$3"
  curl -L --fail --retry 3 --max-time 180 -sS \
    -H 'Accept: application/vnd.github.raw+json' \
    "https://api.github.com/repos/${repo}/git/blobs/${sha}" \
    -o "${target}"
}

download_blob "flclain/StanfordDroneDataset" \
  "c7e056bd7562b866a25e6adaea394666b2e4a8ff" \
  "${SDD_ROOT}/annotations/bookstore/video0/annotations.txt"
download_blob "flclain/StanfordDroneDataset" \
  "56f8f8299cc32e622cc7b1aa2639a888ccdd6249" \
  "${SDD_ROOT}/annotations/bookstore/video1/annotations.txt"
download_blob "flclain/StanfordDroneDataset" \
  "8f0dab37a18dab5b4dfa56cf7032cff1e0fa5d4f" \
  "${SDD_ROOT}/annotations/gates/video0/annotations.txt"
download_blob "flclain/StanfordDroneDataset" \
  "3823d20397246df05f6a47120d3a1673c459fcbe" \
  "${SDD_ROOT}/annotations/gates/video1/annotations.txt"
download_blob "flclain/StanfordDroneDataset" \
  "6df67685fa31bd84786cf6a50ac7d9c355c779a3" \
  "${SDD_ROOT}/annotations/nexus/video0/annotations.txt"
download_blob "flclain/StanfordDroneDataset" \
  "ec03a08a2b225eafff67fdeb0af84715345a70c6" \
  "${SDD_ROOT}/annotations/nexus/video1/annotations.txt"

curl -L --fail --retry 3 --max-time 180 -sS \
  -H 'Accept: application/vnd.github.raw+json' \
  "https://api.github.com/repos/tmralmeida/thor-magni-actions/git/blobs/1abf20f3b7f538ad29c590fab99aaa818d7f89a7" \
  -o "${ROOT}/thor_magni_merged_data.csv"

printf 'Downloaded public external validation assets under %s\n' "${ROOT}"
printf 'Run the preparation and audit commands from docs/EXTERNAL_DATA_SOURCES.md.\n'
