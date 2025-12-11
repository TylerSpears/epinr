#!/usr/bin/bash
set -eou pipefail

N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export ITK_NIFTI_SFORM_PERMISSIVE="${ITK_NIFTI_SFORM_PERMISSIVE:-1}"

# Also allow the user to set the "$RM_TMP_DIR" env var to avoid deleting the
# intermediate files for debugging.
RM_TMP_DIR="${RM_TMP_DIR:-1}"
export RM_TMP_DIR
# Create temporary directory, allow user to override with TMP_DIR env var.
tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_qsiprep_sdc')"}
tmp_dir="$(realpath "$tmp_dir")"
this_script_dir="$(realpath "$(dirname "$0")")"

# p
pe_dir="NULL"
# r
total_readout_time_s="NULL"
# a
anat_type="NULL"
# j
participant_json="NULL"
# t
participant_tsv="NULL"
# s
subj_id="NULL"
# n
session_id="ses-01"
# d
dataset_description="${this_script_dir}/dataset_description.json"
# f
freesurfer_license="NULL"
while getopts "p:r:a:j:t:s:n:d:f:" opt; do
    case $opt in
    p)
        pe_dir="$OPTARG"
        echo "Using PE direction: $pe_dir"
        ;;
    r)
        total_readout_time_s="$OPTARG"
        echo "Using total readout time (s): $total_readout_time_s"
        ;;
    a)
        anat_type="$OPTARG"
        echo "Using anatomical type: $anat_type"
        ;;
    j)
        participant_json="$(realpath "$OPTARG")"
        echo "Using participant JSON: $participant_json"
        ;;
    t)
        participant_tsv="$(realpath "$OPTARG")"
        echo "Using participant TSV: $participant_tsv"
        ;;
    s)
        subj_id="$OPTARG"
        echo "Using subject ID: $subj_id"
        ;;
    n)
        session_id="$OPTARG"
        echo "Using session ID: $session_id"
        ;;
    d)
        dataset_description="$(realpath "$OPTARG")"
        echo "Using dataset description: $dataset_description"
        ;;
    f)
        freesurfer_license="$(realpath "$OPTARG")"
        echo "Using FreeSurfer license: $freesurfer_license"
        ;;
    *)
        echo "Invalid option: $opt"
        return 1
        ;;
    esac
done

if [[ "$freesurfer_license" == "NULL" ]]; then
    freesurfer_license="${FREESURFER_HOME}/.license"
fi
if [[ ! -f "$freesurfer_license" ]]; then
    echo "FreeSurfer license file not found at: $freesurfer_license"
    exit 1
fi
if [[ "$anat_type" == "NULL" ]] || [[ "$participant_json" == "NULL" ]] || [[ "$participant_tsv" == "NULL" ]] || [[ "$subj_id" == "NULL" ]] ||
[[ "$pe_dir" == "NULL" ]]; then
    echo "Must provide anatomical type, participant JSON, participant TSV, and subject ID."
    exit 1
fi
if [[ "$total_readout_time_s" == "NULL" ]] || { [[ "$pe_dir" != "ap" ]] && [[ "$pe_dir" != "pa" ]] ; } ; then
    echo "Currently only support AP phase-encoding direction with specified total readout time."
    exit 1
fi

shift $((OPTIND-1))
anat="$1"
anat_json="$2"
dwi="$3"
dwi_json="$4"
bvecs="$5"
bvals="$6"
out_dir="$7"

mkdir --parents "$tmp_dir"
echo "=========== Temporary Directory $tmp_dir ==================="

if [[ "$anat_type" == "t1w" ]]; then
    bids_anat_type="T1w"
    ignore_type="t2w"
elif [[ "$anat_type" == "t2w" ]]; then
    bids_anat_type="T2w"
    ignore_type=""
else
    echo "Anatomical type must be 't1w' or 't2w'."
    exit 1
fi

# Create temporary directory for a single-subject "dataset".
tmp_dataset_dir="${tmp_dir}/dataset"
mkdir --parents "${tmp_dataset_dir}/${subj_id}/${session_id}"/{"dwi","anat"}
cp -av "$dataset_description" "${tmp_dataset_dir}/dataset_description.json"
# Copy participant JSON and TSV files.
cp -av "$participant_json" "${tmp_dataset_dir}/participants.json"
cp -av "$participant_tsv" "${tmp_dataset_dir}/participants.tsv"
echo "N/A" >> "${tmp_dataset_dir}/README"
dwi_data_dir="${tmp_dataset_dir}/$subj_id/$session_id/dwi"

# Copy DWI and associated files
if [[ "$pe_dir" == "ap" ]]; then
    dwi_base_fname="${subj_id}_${session_id}_dwi"
    mm2hz_sign="1"
    json_phase_encoding_dir="j-"
elif [[ "$pe_dir" == "pa" ]]; then
    dwi_base_fname="${subj_id}_${session_id}_dwi"
    mm2hz_sign="-1"
    json_phase_encoding_dir="j"
fi
# Convert to RAS orientation.
mrconvert "$dwi" -fslgrad "$bvecs" "$bvals" -strides 1,2,3,4 - |
    mrconvert - "${dwi_data_dir}/${dwi_base_fname}.nii.gz" \
    -export_grad_fsl "${dwi_data_dir}/${dwi_base_fname}.bvec" "${dwi_data_dir}/${dwi_base_fname}.bval"
cp -av "$dwi_json" "${dwi_data_dir}/${dwi_base_fname}.json"
# Add total readout time and phase-encoding direction to DWI JSON.
jq --arg trt "$total_readout_time_s" --arg ped "$json_phase_encoding_dir" \
    '. + {TotalReadoutTime: ($trt | tonumber), PhaseEncodingDirection: $ped}' \
    "${dwi_data_dir}/${dwi_base_fname}.json" > "${dwi_data_dir}/tmp.json" && \
    mv "${dwi_data_dir}/tmp.json" "${dwi_data_dir}/${dwi_base_fname}.json"
/usr/bin/cat "${dwi_data_dir}/${dwi_base_fname}.json" | sed 's/\\//g' | jq -Mc '.' >"${dwi_data_dir}/tmp.json" && \
    mv "${dwi_data_dir}/tmp.json" "${dwi_data_dir}/${dwi_base_fname}.json"
# Get output resolution from the DWI's phase encoding direction/axis.
dwi_spacing=($(mrinfo "${dwi_data_dir}/${dwi_base_fname}.nii.gz" -spacing -quiet))
qsi_out_res="${dwi_spacing[1]}"
echo "Using qsiprep output resolution (mm): $qsi_out_res"

# Copy anat image and sidecar JSON.
anat_data_dir="${tmp_dataset_dir}/$subj_id/$session_id/anat"
anat_base_fname="${subj_id}_${session_id}_${bids_anat_type}"
mrconvert "$anat" -strides 1,2,3  "${anat_data_dir}/${anat_base_fname}.nii.gz"
/usr/bin/cat "$anat_json" | sed 's/\\//g' | jq -Mc '.' >"${anat_data_dir}/${anat_base_fname}.json"
# Create temporary working directory for qsiprep.
tmp_qsi_workdir="${tmp_dir}/qsiprep_workdir"
mkdir --parents "$tmp_qsi_workdir"
# Create temporary output dir.
tmp_out_dir="${tmp_dir}/qsiprep_output"
mkdir --parents "$tmp_out_dir"
# --skip-anat-based-spatial-normalization \
docker run -ti --rm \
    -v $(realpath "$tmp_dataset_dir"):/data:ro \
    -v $(realpath "$tmp_out_dir"):/out \
    -v $(realpath "$tmp_qsi_workdir"):/tmp/qsiprep_workdir \
    -v $freesurfer_license:/opt/freesurfer/license.txt \
    --user $(id -u):$(id -g) \
    --ipc host \
    pennlinc/qsiprep:latest \
    /data /out participant --work-dir /tmp/qsiprep_workdir/ \
    --participant-label "$subj_id" --session-id "$session_id" \
    --fs-license-file "/opt/freesurfer/license.txt" \
    --subject-anatomical-reference sessionwise \
    --anat-modality $bids_anat_type \
    --b0-to-t1w-transform Rigid \
    --denoise-method dwidenoise \
    --b0-motion-corr-to first --hmc-transform Rigid \
    --use-syn-sdc error --force-syn \
    --output-resolution $qsi_out_res \
    --denoise-after-combining \
    --hmc-model none --unringing-method none --no-b0-harmonization --b1-biascorrect-stage none \
    --ignore $ignore_type phase sbref flair \
    --nprocs $N_PROCS --omp-nthreads $N_PROCS \
    --notrack --write-graph --stop-on-first-crash --verbose |
    tee "${tmp_dir}/qsiprep_${subj_id}_log.txt"
# Must put the `--subject-anatomical-reference sessionwise` flag to avoid erroring
# out.
out_subj_id="$(echo $subj_id | tr '-' '_')"
out_session_id="$(echo $session_id | tr '-' '_')"
# /qsiprep_1_0_wf/sub_HC001_ses_01_wf/dwi_preproc_ses_01_wf/qsiprep_hmcsdc_wf/sdc_wf/syn_sdc_wf/
sdc_dir="${tmp_qsi_workdir}/qsiprep_1_0_wf/${out_subj_id}_${out_session_id}_wf/dwi_preproc_${out_session_id}_wf/qsiprep_hmcsdc_wf/sdc_wf/syn_sdc_wf/"
sdc_warp="${sdc_dir}/syn/ants_susceptibility0Warp.nii.gz"
qsi_unwarped_b0="${sdc_dir}/unwarp_ref/average_trans.nii"
rm -rfv "$out_dir"
mkdir --parents "$out_dir"
mrconvert "$qsi_unwarped_b0" -strides 1,2,3 "${out_dir}/qsiprep_b0_unwarped.nii.gz"
# Displacement field is a 5D image from ANTs: X x Y x Z x 1 x 3 (directional vectors),
# where the vector components are 0 except in the PE direction (Y here).
mrconvert "$sdc_warp" -strides 1,2,3,4,5 -coord 4 1 -coord 3 0 -axes 0,1,2 \
    "${out_dir}/b0_displacement_field_dir-${pe_dir}_mm.nii.gz"
mrcalc "${out_dir}/b0_displacement_field_dir-${pe_dir}_mm.nii.gz" $qsi_out_res -div \
    $mm2hz_sign -mul \
    $total_readout_time_s -div \
    "${out_dir}/b0_suscept_field_dir-${pe_dir}_hz.nii.gz"
cp -av "${tmp_dir}/qsiprep_${subj_id}_log.txt" "$out_dir"

if [[ "$RM_TMP_DIR" == "1" ]]; then
    rm -rfv "$tmp_dir"
else
    echo "Temporary directory not removed: $tmp_dir"
fi
#applytopup -i ~/mnt/magpie/outputs/mica_mics/derivatives/epinr_fmap_learning/sub-HC001/ses-01/dwi-0/b0.nii.gz -a  ~/mnt/magpie/outputs/mica_mics/derivatives/epinr_fmap_learning/sub-HC001/ses-01/dwi-0/acqparams.txt -x 1 -m jac -n trilinear -t _b0_suscept_field_dir-ap_hz -o applytopup_lin-jac_corrected_b0.nii.gz
