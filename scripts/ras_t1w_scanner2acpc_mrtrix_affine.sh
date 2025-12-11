#!/usr/bin/bash

get_ras_t1w_scanner2acpc_mrtrix_affine() {
    local ras_t1="$1"
    local ras_t1_mask="$2"
    local out_scanner2acpc_affine="$3"
    local out_vox2acpc_affine="$4"
    local ARTHOME="${5:-$ARTHOME}"
    local N_PROCS=${N_PROCS-"$(nproc)"}
    export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
    export OMP_NUM_THREADS=$N_PROCS
    export ITK_NIFTI_SFORM_PERMISSIVE="${ITK_NIFTI_SFORM_PERMISSIVE:-1}"

    # Control verbosity flags.
    # Allow the user to set the "$VERBOSE" env var to debug the commands in this script.
    local VERBOSE="${VERBOSE:-0}"
    if [ "$VERBOSE" = '1' ]; then
        local MRTRIX_VERBOSITY='-info'
        local ANTS_VERBOSITY=1
        local ART_VERBOSITY='-v'
    else
        local MRTRIX_VERBOSITY='-quiet'
        local ANTS_VERBOSITY=0
        local ART_VERBOSITY=''
    fi
    # Also allow the user to set the "$RM_TMP_DIR" env var to avoid deleting the
    # intermediate files for debugging.
    local RM_TMP_DIR="${RM_TMP_DIR:-1}"

    # Requires mrtrix commands, ANTS commands, and ART `acpcdetect`.
    local c
    for c in "mrconvert" "mrcalc" "mrgrid" "transformcompose" "mrtransform" "mrinfo" "mrstats"; do
        if ! command -v "$c" &>/dev/null; then
            echo "Program $c not found. Please install mrtrix3."
            return 1
        fi
    done
    for c in "N4BiasFieldCorrection" "DenoiseImage"; do
        if ! command -v "$c" &>/dev/null; then
            echo "Program $c not found. Please install ANTS."
            return 1
        fi
    done
    if ! command -v "${ARTHOME}/bin/acpcdetect" &>/dev/null; then
        echo "acpcdetect not found. Please install ART/acpcdetect."
        return 1
    fi

    local tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_acpc_align_t1w')"}
    tmp_dir="$(realpath "$tmp_dir")"
    mkdir --parents "$tmp_dir"
    echo "=========== Temporary Directory $tmp_dir ==================="

    local orig_t1="$ras_t1"
    local orig_mask="$ras_t1_mask"
    local t1="$ras_t1"
    local mask="$ras_t1_mask"
    # Check for image values < 0.
    local num_neg_vox
    num_neg_vox="$(mrcalc $MRTRIX_VERBOSITY -nthreads $N_PROCS "$t1" 0 -lt - | mrstats -nthreads $N_PROCS $MRTRIX_VERBOSITY - -output count -ignorezero)"
    local is_negatives
    is_negatives="$(echo "$num_neg_vox > 0" | bc -lq)"
    if [ "$is_negatives" -eq 1 ]; then
        echo "Correcting $num_neg_vox negative voxels in '$t1'"
        local nonneg_t1="${tmp_dir}/nonneg_t1w.nii"
        mrcalc $MRTRIX_VERBOSITY -nthreads $N_PROCS "$t1" 0 -max "$nonneg_t1"
        t1="$nonneg_t1"
    fi

    # Ensure input T1w is in RAS orientation, otherwise error and exit.
    local t1_strides
    t1_strides="$(mrinfo -strides "$t1")"
    if [ "$t1_strides" != "1 2 3" ]; then
        echo "Input T1w image is not in RAS orientation. Please convert to RAS orientation."
        return 1
    fi
    # Resample the brain mask into the T1's space. If they are the same size, then this
    # should just change the strides.
    local ras_mask="${tmp_dir}/ras_brain_mask.nii"
    mrgrid $MRTRIX_VERBOSITY -force \
        "$mask" \
        regrid \
        -template "$t1" \
        -interp nearest \
        "$ras_mask"
    mask="$ras_mask"

    # De-bias and denoise the T1 for cleaner feature matching.
    debias_t1="${tmp_dir}/debias_t1w.nii.gz"
    N4BiasFieldCorrection --verbose $ANTS_VERBOSITY --image-dimensionality 3 \
        --input-image "$t1" \
        --output "$debias_t1"
    denoise_t1="${tmp_dir}/denoise_t1w.nii.gz"
    DenoiseImage --verbose $ANTS_VERBOSITY --image-dimensionality 3 \
        --input-image "$debias_t1" \
        --noise-model Gaussian \
        --output "$denoise_t1"
    t1="$denoise_t1"

    # Mask the T1w image with the brain mask.
    local masked_t1="${tmp_dir}/masked_t1w.nii.gz"
    mrcalc $MRTRIX_VERBOSITY -nthreads $N_PROCS "$t1" "$mask" -mult "$masked_t1"
    t1="$masked_t1"

    # Get rigid transformation to ACPC alignment.
    # Use acpcdetect to perform the AC-PC alignment. See:
    # - <https://github.com/ardekb01/babak_lib>,
    # - <https://www.nitrc.org/projects/art/>, and
    # - B. A. Ardekani. A new approach to symmetric registration of longitudinal structural
    #   MRI of the human brain. J Neurosci Methods, 373:109563, May 2022.
    # acpcdetect requires a .nii file, (not .nii.gz) that is either int16 or uint16.
    local acpcdetect_t1w="${tmp_dir}/acpcdetect_t1w.nii"
    mrconvert $MRTRIX_VERBOSITY -nthreads $N_PROCS -force \
        "$t1" -datatype int16 \
        "$acpcdetect_t1w"
    # acpcdetect always writes files to the current directory, so use pushd and popd to
    # return to the original directory.
    pushd "$tmp_dir" || return 1
    "${ARTHOME}/bin/acpcdetect" $ART_VERBOSITY \
        -nopng \
        -noppm \
        -notxt \
        -no-tilt-correction \
        -output-orient RAS \
        -center-AC \
        -i "$acpcdetect_t1w"
    popd || return 1
    local tilt_correct_vol
    tilt_correct_vol="${tmp_dir}/$(basename "${acpcdetect_t1w}" .nii)_RAS.nii"
    # Extract the tilt-correction affine from acpcdetect, then (slightly) translate the
    # coordinate space such that the subject T1w origin matches the MNI152 template origin.
    # E.g., coordinate (0, 0, 0) is located slightly posterior, superior of the AC in the
    # actual MNI template.
    # See:
    # - C. M. Lacadie, R. K. Fulbright, N. Rajeevan, R. T. Constable, and X. Papademetris,
    # “More accurate Talairach coordinates for neuroimaging using non-linear
    # registration,” NeuroImage, vol. 42, no. 2, pp. 717–725, Aug. 2008,
    # doi: 10.1016/j.neuroimage.2008.04.240.
    # and
    # - <https://bioimagesuiteweb.github.io/webapp/mni2tal.html>

    local tilt_correct_affine="${tmp_dir}/tilt_correct_affine.txt"
    mrinfo $MRTRIX_VERBOSITY -transform "$tilt_correct_vol" >"$tilt_correct_affine"
    read -d '' art_center_ac2acpc_affine_str <<EOF || true
1                 0                 0                 0
0                 1                 0                 2
0                 0                 1                -4
0                 0                 0                 1
EOF
    local art_center_ac2acpc_affine="${tmp_dir}/art_center_ac2acpc_affine.txt"
    echo "$art_center_ac2acpc_affine_str" >"$art_center_ac2acpc_affine"

    transformcompose $MRTRIX_VERBOSITY -force \
        "$art_center_ac2acpc_affine" \
        "$tilt_correct_affine" \
        "$out_vox2acpc_affine"
    # Copy the AC-PC-aligned affine transform to the original T1w image.
    # Replace headers in both the T1 and brain mask.
    local acpc_affine_t1="${tmp_dir}/acpc-affine_t1w.nii"
    local acpc_affine_mask="${tmp_dir}/acpc-affine_brain_mask.nii"
    mrtransform $MRTRIX_VERBOSITY -force \
        "$t1" \
        -replace "$out_vox2acpc_affine" \
        "$acpc_affine_t1"
    mrtransform $MRTRIX_VERBOSITY -force \
        "$mask" \
        -replace "$out_vox2acpc_affine" \
        "$acpc_affine_mask"

    # Calculate the forward affine transform from the original input image space
    # (vox2scanner) to the tilt-corrected space (ACPC space).
    # transformcalc ... header calculates the pull-back affine transform, so invert it to
    # get the push-forward affine.
    transformcalc $MRTRIX_VERBOSITY \
        "$t1" \
        "$acpc_affine_t1" \
        header \
        "${tmp_dir}/acpc2scanner_affine.txt"
    transformcalc $MRTRIX_VERBOSITY -force \
        "${tmp_dir}/acpc2scanner_affine.txt" \
        invert \
        "$out_scanner2acpc_affine"

    if [ "$RM_TMP_DIR" = '1' ]; then
        rm -rf "$tmp_dir"
    fi

    return 0
}
