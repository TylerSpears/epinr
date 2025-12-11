#!/bin/bash
set -eou pipefail

readout_time="0.063054";
treatment_dirs=("/data/VCU_MS_Study/bids/"P_*);
control_dirs=("/data/VCU_MS_Study/bids/"HC_*);
subj_dirs=("${treatment_dirs[@]}" "${control_dirs[@]}");
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
    subj_id="sub-${subj_id}"
    subj_id="$(echo "$subj_id" | tr -d '_' )"
    echo "Processing subject: $subj_id"
    out_dir="/home/tas6hh/mnt/magpie/outputs/results/epinr/dmri/qsiprep/b0_reg_t1w/vcu-ms_${subj_id}"
    mkdir --parents "$out_dir"
    if [[ -f "${out_dir}/b0_suscept_field_dir-ap_hz.nii.gz" ]]; then
        echo "  Subject already processed. Skipping."
        continue
    fi
    tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_vcu_qsiprep_sdc')"}
    tmp_dir="$(realpath "$tmp_dir")"
    mkdir --parents "$tmp_dir"
    t1w_=("$subj_dir"/*_T1_*_[0-9][0-9][0-9].nii.gz)
    t1w="${t1w_[0]}"
    t1w_base=$(basename "$t1w" .nii.gz)
    t1w_json="${subj_dir}/${t1w_base}.json"
    t2w_=("$subj_dir"/*_T2_*_[0-9][0-9][0-9].nii.gz)
    t2w="${t2w_[0]}"
    t2w_base=$(basename "$t2w" .nii.gz)
    t2w_json="${subj_dir}/${t2w_base}.json"
    echo "  T1w: $t1w"
    echo "  T2w: $t2w"
    ap_dwi_=("$subj_dir"/*DKI_AP_*_[0-9][0-9][0-9].nii.gz)
    ap_dwi="${ap_dwi_[0]}"
    if [[ ! -f "$ap_dwi" ]]; then
        echo "  No AP DWI found for subject, skipping."
        continue
    fi
    ap_base=$(basename "$ap_dwi" .nii.gz)
    ap_bval="${subj_dir}/${ap_base}.bval"
    ap_bvec="${subj_dir}/${ap_base}.bvec"
    dwi_json="${subj_dir}/${ap_base}.json"
    mrconvert "$ap_dwi" -fslgrad "$ap_bvec" "$ap_bval" -strides 1,2,3 "${tmp_dir}/dwi_ap.mif"
    dwiextract "${tmp_dir}/dwi_ap.mif" -bzero - |
        mrconvert - -coord 3 0 "${tmp_dir}/b0_ap.mif" || break
    dwiextract "${tmp_dir}/dwi_ap.mif" -no_bzero "${tmp_dir}/dwi_ap_nob0.mif" || break
    mrcat "${tmp_dir}/b0_ap.mif" "${tmp_dir}/b0_ap.mif" "${tmp_dir}/dwi_ap_nob0.mif" - |
        mrconvert - "${tmp_dir}/combined_dwi_ap.nii.gz" \
        -export_grad_fsl "${tmp_dir}/combined_dwi_ap.bvec" "${tmp_dir}/combined_dwi_ap.bval" || break
    N_PROCS=9 TMP_DIR="$tmp_dir" /home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/comparison_methods/qsiprep/run_qsiprep.sh \
        -p "ap" \
        -r $readout_time \
        -a "t1w" \
        -j "/home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/preproc/vcu_ms/participants.json" \
        -t "/home/tas6hh/Projects/mr-inr/notebooks/epinr/dmri/preproc/vcu_ms/participants.tsv" \
        -s "$subj_id" \
        "$t1w" \
        "$t1w_json"  \
        "${tmp_dir}/combined_dwi_ap.nii.gz" \
        "$dwi_json" \
        "${tmp_dir}/combined_dwi_ap.bvec" \
        "${tmp_dir}/combined_dwi_ap.bval" \
        "$out_dir" || break
done

# Apply displacement field to b0 image with applytopup.
treatment_dirs=("/data/VCU_MS_Study/bids/"P_*);
control_dirs=("/data/VCU_MS_Study/bids/"HC_*);
subj_dirs=("${treatment_dirs[@]}" "${control_dirs[@]}");
IFS=$'\n' sorted_subj_dirs=($(sort <<<"${subj_dirs[*]}"));
unset IFS;
for subj_dir in "${sorted_subj_dirs[@]}"; do
    subj_id=$(basename "$subj_dir")
    bids_subj_id="sub-${subj_id}"
    bids_subj_id="$(echo "$bids_subj_id" | tr -d '_' )"
    echo "Processing subject: $subj_id / $bids_subj_id"
    # T1w
    out_dir="/home/tas6hh/mnt/magpie/outputs/results/epinr/dmri/qsiprep/b0_reg_t1w/vcu-ms_${bids_subj_id}"
    # T2w
    # out_dir="/home/tas6hh/mnt/magpie/outputs/results/epinr/dmri/qsiprep/b0_reg_t2w/vcu-ms_${bids_subj_id}"
    in_img_dir="/home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/${subj_id}/dwi-0"
    in_b0="${in_img_dir}/b0.nii.gz"
    acq="${in_img_dir}/acqparams.txt"
    sf="${out_dir}/b0_suscept_field_dir-ap_hz.nii.gz"
    if [[ ! -f "$sf" ]] || [[ ! -f "$in_b0" ]]; then
        echo "  Subject $subj_id not yet processed. Skipping."
        continue
    fi
    out_f="${out_dir}/applytopup_lin-jac_corrected_b0.nii.gz"
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
