#!/bin/bash
set -eou pipefail

N_PROCS=${N_PROCS-"$(nproc)"}
export N_PROCS
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export MRTRIX_NTHREADS=$N_PROCS

# Set up MICA-MICS directory for sending to EPINR code.
raw_root_dir="/home/tas6hh/mnt/magpie/mica_mics"
preproc_dir="/home/tas6hh/mnt/magpie/outputs/mica_mics/derivatives/preproc_dwi"
fs_dir="/home/tas6hh/mnt/magpie/outputs/mica_mics/derivatives/freesurfer"
out_dir="/home/tas6hh/mnt/magpie/outputs/mica_mics/derivatives/epinr_fmap_learning"

# Hard-coded constants for mica-mics.
dataset_name="mica_mics"
ses_id="01"
run_id="1"
total_readout_time_s="0.05282"
# Only take first b0 volume that is the target of the topup correction, i.e. no motion params.
dwi_idx="0"
pe_dir="ap"

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
RIGID_REG_SCRIPT="${SCRIPT_LOOKUP_DIR}/rigid_reg_riders.sh"
SYNTHSTRIP_SCRIPT="${SCRIPT_LOOKUP_DIR}/synthstrip-docker"
# ACPC_ALIGN_SCRIPT="${SCRIPT_LOOKUP_DIR}/ras_t1w_scanner2acpc_mrtrix_affine.sh"
# # shellcheck source=/dev/null
# source "$ACPC_ALIGN_SCRIPT"

for subj_dir in "$preproc_dir"/sub-*/; do
    echo "$subj_dir"
    tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_mica_mics_epinr')"}
    tmp_dir="$(realpath "$tmp_dir")"
    mkdir --parents "$tmp_dir"

    subj_id=$(basename "$subj_dir")
    subj_raw_dir="${raw_root_dir}/${subj_id}/ses-${ses_id}"
    subj_fs_dir="${fs_dir}/${subj_id}_ses-${ses_id}_run-${run_id}"
    subj_topup_dir="${subj_dir}/ses-${ses_id}/dwi/topup"
    topup_fmap="${subj_topup_dir}/topup_suscept_field_hz.nii.gz"
    topup_fmap_coeffs="${subj_topup_dir}/topup_fieldcoef.nii.gz"
    distorted_b0s="${subj_topup_dir}/select_b0s.nii.gz"
    topup_corrected_b0s="${subj_topup_dir}/topup_corrected_b0s.nii.gz"
    t1w="${subj_raw_dir}/anat/${subj_id}_ses-${ses_id}_run-1_T1w.nii.gz"

    subj_out_dir="${out_dir}/${subj_id}/ses-${ses_id}/dwi-${dwi_idx}"
    out_t1w="${subj_out_dir}/t1w.nii.gz"
    out_t1w_mask="${subj_out_dir}/t1w_mask.nii.gz"
    mkdir --parents "$subj_out_dir"
    out_b0="${subj_out_dir}/b0.nii.gz"
    out_topup_b0="${subj_out_dir}/topup_corrected_b0.nii.gz"
    out_b0_mask="${subj_out_dir}/b0_mask.nii.gz"
    out_mni2t1w_warp="${subj_out_dir}/mni2t1w_ants_composite_warp.h5"

    # 1. Copy topup outputs to the subject output directory.
    # Copy topup outputs to the output directory.
    # Copy the acqparams file into the output directory.
    head -n 1 "${subj_topup_dir}/acqparams.txt" > "${subj_out_dir}/acqparams.txt"
    # Same for the movement parameters, which should all be 0s.
    head -n 1 "${subj_topup_dir}/topup_movpar.txt" > "${subj_out_dir}/movpar.txt"
    # Copy fmaps.
    rsync --checksum --copy-links --archive "$topup_fmap" "${subj_out_dir}/topup_suscept_field_hz.nii.gz"
    rsync --checksum --copy-links --archive "$topup_fmap_coeffs" "${subj_out_dir}/topup_fieldcoef.nii.gz"

    # 2. Copy the first b0 volume.
    if [ ! -f "$out_b0" ]; then
        # Delete the mask, if it exists, so it can be re-created.
        rm -vf "$out_b0" "$out_b0_mask"
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
    fi
    if [ ! -f "$out_topup_b0" ]; then
        mrconvert "$topup_corrected_b0s" -coord 3 0 -axes 0,1,2 "$out_topup_b0"
    fi

    # Create temporary b0 mask using bet.
    bet_b0_mask="${tmp_dir}/bet_b0_mask.nii.gz"
    bet \
        "$out_b0" \
        "${tmp_dir}/bet_b0" \
        -m -n -v \
        -F
    # 3. Preproc the anat image and create mask from synthstrip.
    if [[ ! -f "$out_t1w" ]] || [[ ! -f "$out_t1w_mask" ]]; then
        # Delete the outputs, if they exist, so they can be re-created.
        rm -vf "$out_t1w" "$out_t1w_mask"
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
        # Rigid registration of T1 to b0.
        "$RIGID_REG_SCRIPT" \
            -m "$synthstrip_t1w_mask" \
            -f "$bet_b0_mask" \
            -M "$out_t1w_mask" \
            -v \
            "$denoise_t1" \
            "$debias_b0" \
            "$out_t1w"
    fi

    # 4. Create distorted EPI mask.
    if [ ! -f "$out_b0_mask" ]; then
        # Delete the mask, if it exists, so it can be re-created.
        rm -vf "$out_b0_mask"
        # Merge the bet b0 mask and the anat synthstrip mask to create a mask for the
        # distorted b0 image.
        mrgrid "$out_t1w_mask" regrid -template "$out_b0" -interp nearest - |
            mrcalc - "$bet_b0_mask" -or - |
            maskfilter - dilate -npass 3 "$out_b0_mask"
    fi

    # 5. Warp MNI template to the T1w image and save the warps.
    if [ ! -f "$out_mni2t1w_warp" ]; then
        antsRegistration --verbose 1 --random-seed 1 --dimensionality 3 --float 0 \
            --write-composite-transform 1 \
            --collapse-output-transforms 1 \
            --output [ "${tmp_dir}/ants_mni_reg_t1w_" ] \
            --interpolation Linear \
            --use-histogram-matching 1 \
            --winsorize-image-intensities [ 0.005,0.995 ] \
            -x [ "${FSLDIR}/data/standard/MNI152_T1_1mm_brain_mask.nii.gz","$out_t1w_mask" ] \
            --initial-moving-transform [ "${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz","$out_t1w",1 ] \
            --transform Rigid[ 0.1 ] \
            --metric GC[ "${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz","$out_t1w",1,1,Regular,0.25 ] \
            --convergence [ 1000x500x250x100,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox \
            --transform Affine[ 0.1 ] \
            --metric GC[ "${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz","$out_t1w",1,1,Regular,0.25 ] \
            --convergence [ 1000x500x250x100,1e-6,10 ] \
            --shrink-factors 12x8x4x2 \
            --smoothing-sigmas 4x3x2x1vox \
            --transform SyN[ 0.1,3,0 ] \
            --metric CC[ "${FSLDIR}/data/standard/MNI152_T1_1mm.nii.gz","$out_t1w",1,4 ] \
            --convergence [ 100x100x70x50x20,1e-6,10 ] \
            --shrink-factors 10x6x4x2x1 \
            --smoothing-sigmas 5x3x2x1x0vox
        mv -v "${tmp_dir}/ants_mni_reg_t1w_Composite.h5" "$out_mni2t1w_warp"
        # Apply the MNI warp to the Treiber et al, 2016 atlas.
        antsApplyTransforms -d 3 \
            -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S1_T1-avg-atlas.nii.gz \
            -r "${subj_out_dir}/t1w.nii.gz" \
            -t "${subj_out_dir}/mni2t1w_ants_composite_warp.h5" \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
            -n Linear \
            -o "${subj_out_dir}/suscept-atlas_warped-subj_t1w.nii.gz" || break
        antsApplyTransforms -d 3 \
            -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S2_epi-distortion-avg.nii.gz \
            -r "${subj_out_dir}/t1w.nii.gz" \
            -t "${subj_out_dir}/mni2t1w_ants_composite_warp.h5" \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
            -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
            -n Linear \
            -o "${subj_out_dir}/suscept-atlas-warped_suscept-field_dir-ap_mm.nii.gz" || break
    fi
    rm -rvf "$tmp_dir"
done


# dataset_name="mica_mics"
# ses_id="01"
# run_id="1"
# preproc_dir="/home/tas6hh/mnt/magpie/outputs/mica_mics/derivatives/preproc_dwi"
# out_dir="/home/tas6hh/mnt/magpie/outputs/mica_mics/derivatives/epinr_fmap_learning"
# dwi_idx="0"
# pe_dir="ap"
# for subj_dir in "$preproc_dir"/sub-*/; do
#     echo "$subj_dir"
#     subj_id=$(basename "$subj_dir")
#     echo "$subj_id"
#     subj_out_dir="${out_dir}/${subj_id}/ses-${ses_id}/dwi-${dwi_idx}"
#     antsApplyTransforms -d 3 \
#         -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S1_T1-avg-atlas.nii.gz \
#         -r "${subj_out_dir}/t1w.nii.gz" \
#         -t "${subj_out_dir}/mni2t1w_ants_composite_warp.h5" \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
#         -n Linear \
#         -o "${subj_out_dir}/suscept-atlas_warped-subj_t1w.nii.gz" || break
#     antsApplyTransforms -d 3 \
#         -i /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/Treiber.etal.2016_S2_epi-distortion-avg.nii.gz \
#         -r "${subj_out_dir}/t1w.nii.gz" \
#         -t "${subj_out_dir}/mni2t1w_ants_composite_warp.h5" \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_1Warp.nii.gz \
#         -t /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/suscept_atlas/reg_atlas_to_mni/atlas_reg_mni_0GenericAffine.mat  \
#         -n Linear \
#         -o "${subj_out_dir}/suscept-atlas_warped-subj_mm.nii.gz" || break
# done
