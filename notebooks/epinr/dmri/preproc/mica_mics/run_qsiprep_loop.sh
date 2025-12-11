#!/usr/bin/bash

# Run qsiprep with b0 registration to t1w for mica-mics data.
for subj_data_dir in sub-*; do
    subj_id="$(basename "$subj_data_dir")"
    echo "Processing subject: $subj_id"
    subj_dir="/home/tas6hh/mnt/magpie/mica_mics/${subj_id}/ses-01"
    out_dir="/home/tas6hh/mnt/magpie/outputs/results/epinr/dmri/qsiprep/b0_reg_t1w/mica-mics_${subj_id}/ses-01/dwi-0/"
    if [[ -f "${out_dir}/b0_suscept_field_dir-ap_hz.nii.gz" ]]; then
        echo "  Subject already processed. Skipping."
        continue
    fi
    tmp_dir="$(mktemp -d --suffix='_qsiprep_sdc')"
    scratch_dir="${tmp_dir}/mrtrix_scratch"
    mkdir --parents "$scratch_dir"
    export MRTRIX_TMPFILE_DIR="$scratch_dir"
    dwi1_base="${subj_dir}/dwi/${subj_id}_ses-01_acq-b300-11_dir-AP_dwi"
    dwi2_base="${subj_dir}/dwi/${subj_id}_ses-01_acq-b700-41_dir-AP_dwi"
    dwi3_base="${subj_dir}/dwi/${subj_id}_ses-01_acq-b2000-91_dir-AP_dwi"
    # Only keep a single b0 in the first b300 acquisition set.
    mrconvert "${dwi1_base}.nii.gz" -fslgrad "${dwi1_base}.bvec" "${dwi1_base}.bval" \
        -strides 1,2,3,4 \
        "${tmp_dir}/dwi1.mif" || break
    dwiextract "${tmp_dir}/dwi1.mif" -shells 5 "${tmp_dir}/b0_copy.mif" || break
    mrconvert "${dwi2_base}.nii.gz" -fslgrad "${dwi2_base}.bvec" "${dwi2_base}.bval" \
        -strides 1,2,3,4 - |
        dwiextract - -shells 700 "${tmp_dir}/dwi2.mif" || break
    mrconvert "${dwi3_base}.nii.gz" -fslgrad "${dwi3_base}.bvec" "${dwi3_base}.bval" \
        -strides 1,2,3,4 - |
        dwiextract - -shells 2000 "${tmp_dir}/dwi3.mif" || break
    # Combine all dwi data.
    mrcat \
        "${tmp_dir}/b0_copy.mif" "${tmp_dir}/dwi1.mif" "${tmp_dir}/dwi2.mif" "${tmp_dir}/dwi3.mif" - |
        mrconvert - \
        "${tmp_dir}/combined_dwi_ap.nii.gz" \
        -export_grad_fsl "${tmp_dir}/combined_dwi_ap.bvec" "${tmp_dir}/combined_dwi_ap.bval" || break
    N_PROCS=18 TMP_DIR="$tmp_dir" /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/comparison_methods/qsiprep/run_qsiprep.sh \
        -p "ap" \
        -r "0.05282" \
        -a "t1w" \
        -j "/home/tas6hh/mnt/magpie/mica_mics/participants.json" \
        -t "/home/tas6hh/mnt/magpie/mica_mics/participants.tsv" \
        -s "$subj_id" \
        "${subj_dir}/anat/${subj_id}_ses-01_run-1_T1w.nii.gz" \
        "${subj_dir}/anat/${subj_id}_ses-01_run-1_T1w.json"  \
        "${tmp_dir}/combined_dwi_ap.nii.gz" \
        "${dwi1_base}.json" \
        "${tmp_dir}/combined_dwi_ap.bvec" \
        "${tmp_dir}/combined_dwi_ap.bval" \
        "$out_dir" || break
done

# Apply displacement field to b0 image with applytopup.
for subj_data_dir in sub-*; do
    subj_id="$(basename "$subj_data_dir")"
    echo "Processing subject: $subj_id"
    subj_dir="/home/tas6hh/mnt/magpie/mica_mics/${subj_id}/ses-01"
    in_img_dir="/home/tas6hh/mnt/magpie/outputs/mica_mics/derivatives/epinr_fmap_learning/${subj_id}/ses-01/dwi-0/"
    out_dir="/home/tas6hh/mnt/magpie/outputs/results/epinr/dmri/qsiprep/b0_reg_t1w/mica-mics_${subj_id}/ses-01/dwi-0/"
    sf="${out_dir}/b0_suscept_field_dir-ap_hz.nii.gz"
    in_b0="${in_img_dir}/b0.nii.gz"
    acq="${in_img_dir}/acqparams.txt"
    out_f="${out_dir}/applytopup_lin-jac_corrected_b0.nii.gz"
    if [[ ! -f "$sf" ]]; then
        echo "  Suseceptibility field not found for subject $subj_id, skipping."
        continue
    fi
    if [[ -f "$out_f" ]]; then
        echo "  Subject $subj_id already processed. Skipping."
        continue
    fi
    sf="$(echo "$sf" | sed 's/.nii.gz//g' )"
    applytopup --verbose \
        --imain="$in_b0" \
        --datain="$acq" \
        --inindex=1 \
        --topup="$sf" \
        --method=jac \
        --interp=trilinear \
        --out="$out_f" || break
done


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
