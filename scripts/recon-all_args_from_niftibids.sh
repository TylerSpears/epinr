#!/usr/bin/env bash

set -eou pipefail

input_nifti="$1"
input_json_sidecard="$2"

# Requires mrtrix `mrconvert` and `mrinfo` commands, as well as freesurfer's
# `mri_convert`.
if ! command -v mrconvert &>/dev/null; then
    echo "mrconvert not found. Please install mrtrix3."
    exit 1
fi
if ! command -v mrinfo &>/dev/null; then
    echo "mrinfo not found. Please install mrtrix3."
    exit 1
fi
if ! command -v mri_convert &>/dev/null; then
    echo "mri_convert not found. Please install freesurfer."
    exit 1
fi
if ! command -v jq &>/dev/null; then
    echo "jq not found. Please install jq."
    exit 1
fi

tmp_dir="$(mktemp -d --suffix='_niftibids2mgz')"
# Parse the json sidecard and filter out any "n/a" values, as those are invalid.
filter_sidecard="${tmp_dir}/sidecard.json"
/usr/bin/cat "$input_json_sidecard" | jq 'del(..|select(. == "n/a"))' >"$filter_sidecard"
# Take advantage of mrtrix's parsing functionality and read the properties directly from
# the .mif file.
mif="${tmp_dir}/input.mif"
mrconvert -quiet \
    "$input_nifti" -json_import "$filter_sidecard" \
    "$mif"

flags=""
# Check for '-mprage' flag using PulseSequenceType.
pst="$(mrinfo "$mif" -property PulseSequenceType 2>/dev/null)"
if [ "$pst" ]; then
    pst="$(echo "$pst" | sed 's/3D//' | tr -d '[:blank:]' | tr '[:lower:]' '[:upper:]')"
    if [ "$pst" == "MPRAGE" ]; then
        flags="-mprage ${flags}"
    fi
fi

# Check for '-3T' flag with field strength.
f_strength="$(mrinfo "$mif" -property MagneticFieldStrength 2>/dev/null)"
if [ "$f_strength" ] && [ "$f_strength" == "3T" ]; then
    flags="-3T ${flags}"
fi
rm -rf "$tmp_dir"

echo "$flags"
