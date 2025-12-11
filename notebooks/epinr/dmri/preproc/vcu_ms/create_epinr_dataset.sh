#!/bin/bash
set -eou pipefail

N_PROCS=${N_PROCS-"$(nproc)"}
export N_PROCS
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export MRTRIX_NTHREADS=$N_PROCS

# If in a git repo, navigate to the expected location of each script.
SCRIPT_LOOKUP_DIR="${SCRIPT_LOOKUP_DIR:-""}"
if [ -z "${SCRIPT_LOOKUP_DIR}" ]; then
    if [ "$(git rev-parse --is-inside-work-tree 2>/dev/null)" == "true" ]; then
        SCRIPT_LOOKUP_DIR="${GIT_REPO_ROOT:-$(git rev-parse --show-toplevel)}/scripts"
    else
        echo "Unable to find location of script. Exiting."
        return 1
    fi
fi
SYNTHSTRIP_SCRIPT="${SCRIPT_LOOKUP_DIR}/synthstrip-docker"
RIGID_REG_SCRIPT="${SCRIPT_LOOKUP_DIR}/rigid_reg_riders.sh"

# Hard-coded constants for VCU.
ses_id="01"
run_id="1"
# Only take first b0 volume that is the target of the topup correction, i.e. no motion params.
dwi_idx="0"
readout_time="0.063054"

treatment_dirs=("/data/VCU_MS_Study/bids/"P_*)
control_dirs=("/data/VCU_MS_Study/bids/"HC_*)
subj_dirs=("${treatment_dirs[@]}" "${control_dirs[@]}")
# Optional reversing of subject processing order, for ad-hoc parallel job runs.
IFS=$'\n' sorted_subj_dirs=($(sort <<<"${subj_dirs[*]}"));
unset IFS;
reverse() {
    # first argument is the array to reverse
    # second is the output array
    declare -n arr="$1" rev="$2"
    for i in "${arr[@]}"
    do
        rev=("$i" "${rev[@]}")
    done
};
declare -a reversed_subj_dirs;
reverse sorted_subj_dirs reversed_subj_dirs;
for subj_dir in "${sorted_subj_dirs[@]}"; do
# for subj_dir in "${reversed_subj_dirs[@]}"; do
    subj_id=$(basename "$subj_dir")
    echo "Processing subject: $subj_id"
    out_dir="/home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/preproc_dwi/${subj_id}"
    mkdir --parents "$out_dir"

    tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_vcu_epinr')"}
    tmp_dir="$(realpath "$tmp_dir")"
    mkdir --parents "$tmp_dir"

    t1w_=("$subj_dir"/*_T1_*_[0-9][0-9][0-9].nii.gz)
    t1w="${t1w_[0]}"
    t2w_=("$subj_dir"/*_T2_*_[0-9][0-9][0-9].nii.gz)
    t2w="${t2w_[0]}"
    flair_=("$subj_dir"/*_FLAIR_*_[0-9][0-9][0-9].nii.gz)
    flair="${flair_[0]}"
    if [[ ! -f "$t1w" || ! -f "$t2w" || ! -f "$flair" ]]; then
        echo "  Missing one or more structural scan files. Skipping subject."
        continue
    fi
    echo "  T1w: $t1w"
    echo "  T2w: $t2w"
    echo "  FLAIR: $flair"

    # File that indicates the preprocessing was already completed.
    indicator_f="${out_dir}/dwi_mask.nii.gz"
    if [[ ! -f "$indicator_f" ]]; then

        ap_dwi_=("$subj_dir"/*DKI_AP_*_[0-9][0-9][0-9].nii.gz)
        ap_dwi="${ap_dwi_[0]}"
        ap_base=$(basename "$ap_dwi" .nii.gz)
        ap_bval="${subj_dir}/${ap_base}.bval"
        ap_bvec="${subj_dir}/${ap_base}.bvec"
        # dwi_json="${subj_dir}/${ap_base}.json"

        pa_dwi_=("$subj_dir"/*DKI_PA_*_[0-9][0-9][0-9].nii.gz)
        pa_dwi="${pa_dwi_[0]}"
        pa_base=$(basename "$pa_dwi" .nii.gz)
        pa_bval="${subj_dir}/${pa_base}.bval"
        pa_bvec="${subj_dir}/${pa_base}.bvec"

        if [[ ! -f "$ap_dwi" || ! -f "$pa_dwi" ]]; then
            echo "  Missing one or more required DWI files. Skipping subject."
            continue
        fi

        echo "  AP DWI: $ap_dwi"
        echo "  PA DWI: $pa_dwi"

        # Create a slspec file for the DWI data.
        dwi_size=($(mrinfo $ap_dwi -size -quiet))
        n_dwi_slices="${dwi_size[2]}"
        if [[ "$n_dwi_slices" == "81" ]]; then
            echo "  Detected 81 slices in DWI. Using slspec for 81 slices."
            slspec_f="$(realpath ./slspec.txt)"
        elif [[ "$n_dwi_slices" == "60" ]]; then
            echo "  Detected 60 slices in DWI. Using slspec for 60 slices."
            slspec_f="$(realpath ./slspec_60.txt)"
        else
            echo "  Unexpected number of slices ($n_dwi_slices) in DWI. Skipping subject."
            continue
        fi

        ap_mif="${tmp_dir}/dwi_ap.mif"
        pa_mif="${tmp_dir}/dwi_pa.mif"
        mrconvert "$ap_dwi" -fslgrad "$ap_bvec" "$ap_bval" -strides 1,2,3 "$ap_mif"
        mrconvert "$pa_dwi" -fslgrad "$pa_bvec" "$pa_bval" -strides 1,2,3 "$pa_mif"

        ras_t1="${tmp_dir}/ras_t1w.nii.gz"
        mrconvert "$t1w" -strides 1,2,3 "$ras_t1"
        debias_t1="${tmp_dir}/debias_t1w.nii.gz"
        N4BiasFieldCorrection --verbose 0 --image-dimensionality 3 \
            --input-image "$ras_t1" \
            --output "$debias_t1"
        denoise_t1="${tmp_dir}/denoise_t1w.nii.gz"
        DenoiseImage --verbose 0 --image-dimensionality 3 \
            --input-image "$debias_t1" \
            --noise-model Gaussian \
            --output "$denoise_t1"
        # Mask the T1w with synthstrip.
        synthstrip_t1w_mask="${tmp_dir}/synthstrip_t1w_mask.nii.gz"
        "$SYNTHSTRIP_SCRIPT" \
            --image "$denoise_t1" \
            --mask "$synthstrip_t1w_mask"

        ./preproc_dwi.sh \
            -v \
            -b 3 \
            "$ap_mif" "$pa_mif" \
            "$readout_time" \
            "$slspec_f" \
            "$t1w" \
            "$synthstrip_t1w_mask" \
            "$out_dir"
        rm -f "$ras_t1" "$debias_t1" "$denoise_t1" "$synthstrip_t1w_mask" \
            "$ap_mif" "$pa_mif"
    else
        echo "  Preprocessing already completed for this subject. Skipping."
    fi

    # Change output dir to the EPINR-specific structured folders.
    subj_topup_dir="${out_dir}/topup"
    out_dir="/home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/${subj_id}/dwi-${dwi_idx}"
    mkdir --parents "$out_dir"

    # Inputs for the fmap learning, taken from the preproc stage.
    topup_fmap="${subj_topup_dir}/topup_suscept_field_hz.nii.gz"
    topup_fmap_coeffs="${subj_topup_dir}/topup_fieldcoef.nii.gz"
    distorted_b0s="${subj_topup_dir}/select_b0s.nii.gz"
    topup_corrected_b0s="${subj_topup_dir}/topup_corrected_b0s.nii.gz"
    # Outputs
    out_t1w="${out_dir}/t1w.nii.gz"
    out_t1w_mask="${out_dir}/t1w_mask.nii.gz"
    out_t2w="${out_dir}/t2w.nii.gz"
    out_flair="${out_dir}/flair.nii.gz"
    out_b0="${out_dir}/b0.nii.gz"
    out_topup_b0="${out_dir}/topup_corrected_b0.nii.gz"
    out_b0_mask="${out_dir}/b0_mask.nii.gz"
    out_mni2t1w_warp="${out_dir}/mni2t1w_ants_composite_warp.h5"
    if [[ ! -f "$out_topup_b0" ]]; then
        head -n 1 "${subj_topup_dir}/acqparams.txt" > "${out_dir}/acqparams.txt"
        # Same for the movement parameters, which should all be 0s.
        head -n 1 "${subj_topup_dir}/topup_movpar.txt" > "${out_dir}/movpar.txt"
        # Copy fmaps.
        rsync --checksum --copy-links --archive "$topup_fmap" "${out_dir}/topup_suscept_field_hz.nii.gz"
        rsync --checksum --copy-links --archive "$topup_fmap_coeffs" "${out_dir}/topup_fieldcoef.nii.gz"

        # Copy the first b0 volume and delete the mask, if it exists, so it can be re-created.
        rm -vf "$out_b0" "$out_b0_mask" "$out_topup_b0"
        # Debias the b0 image.
        # Create fsl b-value files for a single b0 volume. Then combine two b0s to make a fake
        # DWI, which some mrtrix commands require.
        printf '%s\n' 0 0 0 > "${tmp_dir}/b0.bvec"
        echo "0" > "${tmp_dir}/b0.bval"
        mrconvert "$distorted_b0s" -coord 3 0 -axes 0,1,2 - |
            mrconvert - -fslgrad "${tmp_dir}/b0.bvec" "${tmp_dir}/b0.bval" "${tmp_dir}/b0_1.mif"
        dwicat "${tmp_dir}/b0_1.mif" "${tmp_dir}/b0_1.mif" "${tmp_dir}/mrtrix_b0.mif"
        debias_b0="${tmp_dir}/debias_b0.nii.gz"
        # Mrtrix bias correct with default adult human parameters.
        dwibiascorrect ants "${tmp_dir}/mrtrix_b0.mif" \
            -ants.b ["100","3"] \
            -ants.c ["1000","0.0"] \
            -ants.s "4" \
            "$debias_b0"
        # Only take one corrected b0 volume.
        mrconvert "$debias_b0" -coord 3 0 -axes 0,1,2 "$out_b0"
        # mrconvert "$distorted_b0s" -coord 3 0 -axes 0,1,2 "$out_b0"
        mrconvert "$topup_corrected_b0s" -coord 3 0 -axes 0,1,2 "$out_topup_b0"
    else
        echo "  Subject already has topup corrected b0. Skipping copy."
    fi

    bet_b0_mask="${tmp_dir}/bet_b0_mask.nii.gz"
    bet \
        "$out_b0" \
        "${tmp_dir}/bet_b0" \
        -m -n -v \
        -F

    # Preproc the anatomical images and register to the distorted b0.
    if [[ ! -f "$out_t1w" || ! -f "$out_t1w_mask" || ! -f "$out_t2w" || ! -f "$out_flair" ]]; then
        # Delete the outputs, if they exist, so they can be re-created.
        rm -vf "$out_t1w" "$out_t1w_mask" "$out_t2w" "$out_flair"

        # De-bias and denoise the T1 for cleaner feature matching in both rigid alignment
        # and distortion correction.
        ras_t1="${tmp_dir}/ras_t1w.nii.gz"
        mrconvert "$t1w" -strides 1,2,3 "$ras_t1"
        debias_t1="${tmp_dir}/debias_t1w.nii.gz"
        N4BiasFieldCorrection --verbose 0 --image-dimensionality 3 \
            --input-image "$ras_t1" \
            --output "$debias_t1"
        denoise_t1="${tmp_dir}/denoise_t1w.nii.gz"
        DenoiseImage --verbose 0 --image-dimensionality 3 \
            --input-image "$debias_t1" \
            --noise-model Gaussian \
            --output "$denoise_t1"
        # Mask the T1w with synthstrip.
        synthstrip_t1w_mask="${tmp_dir}/synthstrip_t1w_mask.nii.gz"
        "$SYNTHSTRIP_SCRIPT" \
            --image "$denoise_t1" \
            --mask "$synthstrip_t1w_mask"
        # Register the processed T1 to the b0 image.
        # Debias b0.
        debias_b0="${tmp_dir}/debias_b0.nii.gz"
        N4BiasFieldCorrection --verbose 0 --image-dimensionality 3 \
            --input-image "$out_b0" \
            --output "$debias_b0"
        # Rigid registration of masked T1 to b0.
        masked_debias_t1="${tmp_dir}/masked_t1w.nii.gz"
        mrcalc "$denoise_t1" "$synthstrip_t1w_mask" -mult "$masked_debias_t1"
        affine_t12b0="${tmp_dir}/t1w2b0_rigid_affine.mat"
        "$RIGID_REG_SCRIPT" \
            -t "CC" \
            -m "$synthstrip_t1w_mask" \
            -f "$bet_b0_mask" \
            -M "$out_t1w_mask" \
            -A "$affine_t12b0" \
            -v \
            "$masked_debias_t1" \
            "$debias_b0" \
            "${tmp_dir}/masked_t1_reg_b0.nii.gz"
        # Apply the transform to the original T1w to get final outputs. Use the previous
        # output image as the reference to keep the same spacing as the input t1.
        antsApplyTransforms --verbose 0 \
            --dimensionality 3 \
            --input "$denoise_t1" \
            --transform "$affine_t12b0" \
            --output "$out_t1w" \
            --reference-image "${tmp_dir}/masked_t1_reg_b0.nii.gz" \
            --interpolation Linear \
            --default-value 0

        # Debias + denoise T2w, then register to the T1w already in b0 orientation.
        ras_t2="${tmp_dir}/ras_t2w.nii.gz"
        mrconvert "$t2w" -strides 1,2,3 "$ras_t2"
        debias_t2="${tmp_dir}/debias_t2w.nii.gz"
        N4BiasFieldCorrection --verbose 0 --image-dimensionality 3 \
            --input-image "$ras_t2" \
            --output "$debias_t2"
        denoise_t2="${tmp_dir}/denoise_t2w.nii.gz"
        DenoiseImage --verbose 0 --image-dimensionality 3 \
            --input-image "$debias_t2" \
            --noise-model Gaussian \
            --output "$denoise_t2"
        # Rigid registration of T2 to T1.
        # No mask for the t2w, just the t1.
        # Use mutual information for speed, no large risk of accuracy loss.
        "$RIGID_REG_SCRIPT" \
            -t "MI" -p "1,32,Regular,0.5" \
            -v \
            "$denoise_t2" \
            "$out_t1w" \
            "$out_t2w"

        # Debias + denoise FLAIR, then register to the T2w already in b0 orientation.
        ras_flair="${tmp_dir}/ras_flair.nii.gz"
        mrconvert "$flair" -strides 1,2,3 "$ras_flair"
        debias_flair="${tmp_dir}/debias_flair.nii.gz"
        N4BiasFieldCorrection --verbose 0 --image-dimensionality 3 \
            --input-image "$ras_flair" \
            --output "$debias_flair"
        denoise_flair="${tmp_dir}/denoise_flair.nii.gz"
        DenoiseImage --verbose 0 --image-dimensionality 3 \
            --input-image "$debias_flair" \
            --noise-model Gaussian \
            --output "$denoise_flair"
        # Rigid registration of FLAIR to T2.
        # No mask for the flair, just the t2, which just uses the t1w mask.
        "$RIGID_REG_SCRIPT" \
            -t "MI" -p "1,32,Regular,0.5" \
            -v \
            "$denoise_flair" \
            "$out_t2w" \
            "$out_flair"
    else
        echo "  Subject already has preprocessed anatomical images. Skipping."
    fi

    # Create distorted EPI mask.
    if [ ! -f "$out_b0_mask" ]; then
        # Delete the mask, if it exists, so it can be re-created.
        rm -vf "$out_b0_mask"
        # Merge the bet b0 mask and the anat synthstrip mask to create a mask for the
        # distorted b0 image.
        mrgrid "$out_t1w_mask" regrid -template "$out_b0" -interp nearest - |
            mrcalc - "$bet_b0_mask" -or - |
            maskfilter - dilate -npass 3 "$out_b0_mask"
    else
        echo "  Subject already has distorted b0 mask. Skipping."
    fi

    out_mni2t1w_warp="${out_dir}/mni2t1w_ants_composite_warp.h5"
    # 5. Warp MNI template to the T1w image and save the warps.
    if [ ! -f "$out_mni2t1w_warp" ]; then
        masked_t1="${tmp_dir}/masked_t1w_for_mni.nii.gz"
        mrcalc "$out_t1w" "$out_t1w_mask" -mult "$masked_t1"
        antsRegistration --verbose 1 --random-seed 1 --dimensionality 3 --float 0 \
            --write-composite-transform 1 \
            --collapse-output-transforms 1 \
            --output [ "${tmp_dir}/ants_mni_reg_t1w_","${tmp_dir}/ants_mni_reg_t1w_warped.nii.gz","${tmp_dir}/ants_t1w_reg_mni_warped.nii.gz" ] \
            --interpolation Linear \
            --use-histogram-matching 1 \
            --winsorize-image-intensities [ 0.005,0.995 ] \
            -x [ "$out_t1w_mask","${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz" ] \
            --initial-moving-transform [ "$masked_t1","${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz",1 ] \
            --transform Rigid[ 0.1 ] \
            --metric GC[ "$masked_t1","${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz",1,1,Regular,0.75 ] \
            --convergence [ 1000x500x250x100,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox \
            --transform Affine[ 0.1 ] \
            --metric GC[ "$masked_t1","${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz",1,1,Regular,0.75 ] \
            --convergence [ 1000x500x250x100,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox \
            --transform SyN[ 0.1,3,0 ] \
            --metric CC[ "$masked_t1","${FSLDIR}/data/standard/MNI152_T1_1mm_brain.nii.gz",1,4,Regular,0.75 ] \
            --convergence [ 100x100x70x50x20,1e-6,10 ] \
            --shrink-factors 10x6x4x2x1 \
            --smoothing-sigmas 5x3x2x1x0vox
        mv -v "${tmp_dir}/ants_mni_reg_t1w_Composite.h5" "$out_mni2t1w_warp"

        # Apply the MNI warp to the Treiber et al, 2016 atlas.
        antsApplyTransforms -d 3 \
            -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S1_T1-avg-atlas.nii.gz \
            -r "${out_dir}/t1w.nii.gz" \
            -t "${out_dir}/mni2t1w_ants_composite_warp.h5" \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
            -n Linear \
            -o "${out_dir}/suscept-atlas_warped-subj_t1w.nii.gz" || break
        antsApplyTransforms -d 3 \
            -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S2_epi-distortion-avg.nii.gz \
            -r "${out_dir}/t1w.nii.gz" \
            -t "${out_dir}/mni2t1w_ants_composite_warp.h5" \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
            -n Linear \
            -o "${out_dir}/suscept-atlas-warped_suscept-field_dir-ap_mm.nii.gz" || break
    else
        echo "  Subject already has MNI to T1w warp. Skipping."
    fi
    rm -rfv "$tmp_dir"
done


# treatment_dirs=("/data/VCU_MS_Study/bids/"P_*)
# control_dirs=("/data/VCU_MS_Study/bids/"HC_*)
# subj_dirs=("${treatment_dirs[@]}" "${control_dirs[@]}")
# dwi_idx=0
# # Optional reversing of subject processing order, for ad-hoc parallel job runs.
# IFS=$'\n' sorted_subj_dirs=($(sort <<<"${subj_dirs[*]}"));
# unset IFS;
# for subj_dir in "${sorted_subj_dirs[@]}"; do
# # for subj_dir in "${reversed_subj_dirs[@]}"; do
#     subj_id=$(basename "$subj_dir")
#     echo "Processing subject: $subj_id"
#     out_dir="/home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/${subj_id}/dwi-${dwi_idx}"
#     if [[ ! -f "${out_dir}/t1w.nii.gz" ]]; then
#         echo "Skipping $subj_id"
#         continue
#     fi
#     # Apply the MNI warp to the Treiber et al, 2016 atlas.
#     antsApplyTransforms -d 3 \
#         -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S1_T1-avg-atlas.nii.gz \
#         -r "${out_dir}/t1w.nii.gz" \
#         -t "${out_dir}/mni2t1w_ants_composite_warp.h5" \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
#         -n Linear \
#         -o "${out_dir}/suscept-atlas_warped-subj_t1w.nii.gz" || break
#     antsApplyTransforms -d 3 \
#         -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S2_epi-distortion-avg.nii.gz \
#         -r "${out_dir}/t1w.nii.gz" \
#         -t "${out_dir}/mni2t1w_ants_composite_warp.h5" \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
#         -n Linear \
#         -o "${out_dir}/suscept-atlas_warped-subj_mm.nii.gz" || break
# done
