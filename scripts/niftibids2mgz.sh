#!/usr/bin/env bash

set -eou pipefail

input_nifti="$1"
input_json_sidecard="$2"
output_mgz="$3"

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
    "$mif" 2>&1
# Check for image values < 0.
num_neg_vox="$(mrcalc -q "$mif" 0 -lt - | mrstats -q - -output count -ignorezero)"
echo "Number of negative voxels in the input T1: $num_neg_vox"
is_negatives="$(echo "$num_neg_vox > 0" | bc -lq)"
if [ "$is_negatives" -eq 1 ]; then
    echo "Correcting $num_neg_vox negative voxels in '$input_nifti'"
    nonneg_t1="${tmp_dir}/nonneg_t1w.mif"
    mrcalc -q "$mif" 0 -max "$nonneg_t1"
    mif="$nonneg_t1"
fi

# Output freesurfer volume to file, then iteratively add properties onto it.
# Use mrconvert instead of mri_convert to create the .mgz in case the input nifti has
# an invalid header, then mrtrix will default to a reasonable affine.
mrconvert -quiet -force "$mif" "$output_mgz"
# mri_convert "$input_nifti" "$output_mgz"

# There are four properties that may potentially be set:
# 1. TR in milliseconds
# 2. TE in milliseconds
# 3. TI in milliseconds
# 4. flip angle in radians

# If a property is not found in the json sidecard, then it will not be set in the mgz
# file.
# <https://surfer.nmr.mgh.harvard.edu/fswiki/mri_convert>
# Field names and units should be defined by the BIDS standard:
# See <https://bids-specification.readthedocs.io/en/stable/modality-specific-files/magnetic-resonance-imaging-data.html>
# for the BIDS standard, and
# <https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/MghFormat> for the freesurfer
# mgh format.

# TR
tr_sec="$(mrinfo "$mif" -property RepetitionTime 2>/dev/null)"
# Validate the property is a number.
if serr=$(echo "$tr_sec + 0" | bc -lq 2>&1 >/dev/null) && test -z "$serr"; then
    is_valid="true"
else
    is_valid="false"
fi
if
    [ "$tr_sec" ] && [ "$is_valid" = "true" ]
then
    tr_ms="$(echo "scale=4; $tr_sec * 1000" | bc -lq)"
    echo "Setting TR to $tr_ms ms."
    mri_convert -tr "$tr_ms" "$output_mgz" "$output_mgz"
fi

# TE
te_sec="$(mrinfo "$mif" -property EchoTime 2>/dev/null)"
# Validate the property is a number.
if serr=$(echo "$te_sec + 0" | bc -lq 2>&1 >/dev/null) && test -z "$serr"; then
    is_valid="true"
else
    is_valid="false"
fi
if [ "$te_sec" ] && [ "$is_valid" = "true" ]; then
    te_ms="$(echo "scale=4; $te_sec * 1000" | bc -lq)"
    echo "Setting TE to $te_ms ms."
    mri_convert -te "$te_ms" "$output_mgz" "$output_mgz"
fi

# TI
ti_sec="$(mrinfo "$mif" -property InversionTime 2>/dev/null)"
# Validate the property is a number.
if serr=$(echo "$ti_sec + 0" | bc -lq 2>&1 >/dev/null) && test -z "$serr"; then
    is_valid="true"
else
    is_valid="false"
fi
if [ "$ti_sec" ]; then
    ti_ms="$(echo "scale=4; $ti_sec * 1000" | bc -lq)"
    echo "Setting TI to $ti_ms ms."
    mri_convert -TI "$ti_ms" "$output_mgz" "$output_mgz"
fi

# Flip-angle
flip_deg="$(mrinfo "$mif" -property FlipAngle 2>/dev/null)"
# Validate the property is a number.
if serr=$(echo "$flip_deg + 0" | bc -lq 2>&1 >/dev/null) && test -z "$serr"; then
    is_valid="true"
else
    is_valid="false"
fi
if [ "$flip_deg" ] && [ "$is_valid" = "true" ]; then
    # Use pi = 4 * atan(1) for better precision.
    # <https://org.coloradomesa.edu/~mapierce2/bc/>
    flip_rad="$(echo "scale=8; $flip_deg * (a(1) / 45)" | bc -lq)"
    echo "Setting flip angle to $flip_rad rads."
    mri_convert -flip_angle "$flip_rad" "$output_mgz" "$output_mgz"
fi

# Two additional properties that may be set with MGH tags, but may not be pulled into
# freesurfer's recon-all:
# 1. MagneticFieldStrength (or "MGH_TAG_FIELDSTRENGTH" if editing in mrtrix)
# 2. PhaseEncodingDirection (or "MGH_TAG_PEDIR" if editing in mrtrix)

# Phase encoding direction.
# Check the PE is valid (i, j, k, or any inverses).
pedir="$(mrinfo "$mif" -property PhaseEncodingDirection 2>/dev/null)"
if [ "$pedir" ] && [[ "$pedir" =~ ^-?[ijk]$ ]]; then
    echo "Setting phase encoding direction to $pedir."
    mrconvert -quiet \
        "$output_mgz" \
        -set_property MGH_TAG_PEDIR "$pedir" \
        "${tmp_dir}/tmp.mgz" 2>&1

    cp "${tmp_dir}/tmp.mgz" "$output_mgz"
    rm -f "${tmp_dir}/tmp.mgz"
fi

# Field strength
f_strength="$(mrinfo "$mif" -property MagneticFieldStrength 2>/dev/null)"
if [ "$f_strength" ]; then
    f_strength_v="$(echo "$f_strength" | sed 's/T//')"
    echo "Setting field strength to $f_strength_v T."
    mrconvert -quiet \
        "$output_mgz" \
        -set_property MGH_TAG_FIELDSTRENGTH $f_strength_v \
        "${tmp_dir}/tmp.mgz" 2>&1
    cp "${tmp_dir}/tmp.mgz" "$output_mgz"
    rm -f "${tmp_dir}/tmp.mgz"
fi

rm -rfv "$tmp_dir"
