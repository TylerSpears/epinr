#!/usr/bin/bash

set -eou pipefail

###################################################################
### Input arguments
## Optional arguments
VERBOSE="0"
denoise_window="5"
n_pe_b0s="1"
eddy_num_vox_hp="1000"
anat_label="t1w"
ap_fmap_b0="NULL"
pa_fmap_b0="NULL"
while getopts "vw:b:x:a:f:g:" opt; do
    case $opt in
    v)
        echo "Verbose: $opt"
        VERBOSE="1"
        ;;
    w)
        echo "dwidenoise window size: $opt, argument: $OPTARG"
        denoise_window="$OPTARG"
        ;;
    b)
        echo "Number of b0s in PE direction: $opt, argument: $OPTARG"
        n_pe_b0s="$OPTARG"
        ;;
    x)
        echo "Eddy number of voxels for GP hyperparam fitting: $opt, argument: $OPTARG"
        eddy_num_vox_hp="$OPTARG"
        ;;
    a)
        echo "Anatomical image name: $opt, argument: $OPTARG"
        anat_label="$OPTARG"
        ;;
    f)
        echo "Override AP b=0 with a .mif image for topup: $opt, argument: $OPTARG"
        ap_fmap_b0="$OPTARG"
        ;;
    g)
        echo "Override PA b=0 with a .mif image for topup: $opt, argument: $OPTARG"
        pa_fmap_b0="$OPTARG"
        ;;
    \?)
        echo "Invalid option: $opt"
        return 1
        ;;
    *)
        echo "Invalid option: $opt"
        return 1
        ;;
    esac
done
## Positional mandatory arguments
ap_dwi_mif="${*:OPTIND:1}"
pa_dwi_mif="${*:OPTIND+1:1}"
readout_time="${*:OPTIND+2:1}"
json_sidecar="${*:OPTIND+3:1}"
anat_f="${*:OPTIND+4:1}"
anat_mask_f="${*:OPTIND+5:1}"
output_dir="${*:OPTIND+6:1}"
echo "AP DWI .mif file: $ap_dwi_mif"
echo "PA DWI .mif file: $pa_dwi_mif"
echo "Total readout time (in seconds): $readout_time"
echo "JSON sidecar file: $json_sidecar"
echo "Anatomical volume file: $anat_f"
echo "Anatomical mask file: $anat_mask_f"
echo "Output directory: $output_dir"

### Variables set by env vars.
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export ITK_NIFTI_SFORM_PERMISSIVE="${ITK_NIFTI_SFORM_PERMISSIVE:-1}"
export MRTRIX_NTHREADS=$N_PROCS
export FSLMULTIFILEQUIT=TRUE
export FSLOUTPUTTYPE=NIFTI_GZ
# Control verbosity flags.
# Allow the user to set the "$VERBOSE" env var to debug the commands in this script.

if [ "$VERBOSE" = '1' ]; then
    MRTRIX_VERBOSITY='-info'
    MRTRIX_LOGLEVEL=2
    ANTS_VERBOSITY=1
    FSL_VERBOSITY="--verbose"
else
    MRTRIX_VERBOSITY='-quiet'
    MRTRIX_LOGLEVEL=0
    ANTS_VERBOSITY=0
    FSL_VERBOSITY=""
fi
export MRTRIX_LOGLEVEL
export MRTRIX_VERBOSITY
# Also allow the user to set the "$RM_TMP_DIR" env var to avoid deleting the
# intermediate files for debugging.
RM_TMP_DIR="${RM_TMP_DIR:-1}"
export RM_TMP_DIR

tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_preproc_dwi')"}
tmp_dir="$(realpath "$tmp_dir")"
mkdir --parents "$tmp_dir"
scratch_dir="${tmp_dir}/mrtrix_scratch"
mkdir --parents "$scratch_dir"
export MRTRIX_TMPFILE_DIR="$scratch_dir"
echo "=========== Temporary Directory $tmp_dir ==================="

########################################################################
### Preproc pipeline
ap_dwi="$ap_dwi_mif"
pa_dwi="$pa_dwi_mif"
anat="$anat_f"
anat_mask="$anat_mask_f"

# Move all volumes to RAS orientation.
ras_ap_dwi="${tmp_dir}/ras_ap_dwi.mif"
ras_pa_dwi="${tmp_dir}/ras_pa_dwi.mif"
ras_anat="${tmp_dir}/ras_anat.nii.gz"
ras_anat_mask="${tmp_dir}/ras_anat_mask.nii.gz"
if [[ "$ap_dwi" != "NULL" ]]; then
    mrconvert "$ap_dwi" -strides 1,2,3,4 "$ras_ap_dwi"
else
    echo "No AP DWIs provided, will use AP b=0 for topup only"
    ras_ap_dwi="NULL"
fi
if [[ "$pa_dwi" != "NULL" ]]; then
    mrconvert "$pa_dwi" -strides 1,2,3,4 "$ras_pa_dwi"
else
    echo "No PA DWIs provided, will use PA b=0 for topup only"
    ras_pa_dwi="NULL"
fi
mrconvert "$anat" -strides 1,2,3 "$ras_anat"
mrconvert "$anat_mask" -strides 1,2,3 "$ras_anat_mask"
ap_dwi="$ras_ap_dwi"
pa_dwi="$ras_pa_dwi"
anat="$ras_anat"
anat_mask="$ras_anat_mask"
# Clean and get the absolute path to the json sidecar.
tmp_json_sidecar="${tmp_dir}/json_sidecar.json"
/usr/bin/cat "$json_sidecar" | sed 's/\\//g' | jq -Mc '.' >"$tmp_json_sidecar"
json_sidecar="$(realpath "$tmp_json_sidecar")"

## 1. Denoise DWIs
pre="10"
denoise_dwi="${tmp_dir}/${pre}_denoise_dwi.mif"
# Concat AP and PA for denoising, then split afterwards.
if [[ "$ap_dwi" != "NULL" ]] && [[ "$pa_dwi" != "NULL" ]]; then
    ap_dwi_size=($(mrinfo $ap_dwi -size -quiet))
    n_ap_dwi="${ap_dwi_size[3]}"
    mrcat "$ap_dwi" "$pa_dwi" -axis 3 - | dwidenoise \
        - \
        -extent $denoise_window \
        -datatype float64 \
        "$denoise_dwi"
    rm -v "$ap_dwi" "$pa_dwi"
    # Split back to AP-PA.
    denoise_ap_dwi="${tmp_dir}/${pre}_denoised_ap_dwi.mif"
    denoise_pa_dwi="${tmp_dir}/${pre}_denoised_pa_dwi.mif"
    mrconvert "$denoise_dwi" -datatype float32 -coord 3 0:$((n_ap_dwi - 1)) "$denoise_ap_dwi"
    mrconvert "$denoise_dwi" -datatype float32 -coord 3 $n_ap_dwi:end "$denoise_pa_dwi"
    rm -v "$denoise_dwi"
    ap_dwi="$denoise_ap_dwi"
    pa_dwi="$denoise_pa_dwi"
elif [[ "$ap_dwi" == "NULL" ]]; then
    echo "AP DWIs not given, denoising only PA DWIs"
    dwidenoise \
        "$pa_dwi" \
        -extent $denoise_window \
        -datatype float64 \
        "$denoise_dwi"
    denoise_pa_dwi="${tmp_dir}/${pre}_denoised_pa_dwi.mif"
    mrconvert "$denoise_dwi" -datatype float32 "$denoise_pa_dwi"
    rm -v "$denoise_dwi"
    pa_dwi="$denoise_pa_dwi"
elif [[ "$pa_dwi" == "NULL" ]]; then
    echo "PA DWIs not given, denoising only AP DWIs"
    dwidenoise \
        "$ap_dwi" \
        -extent $denoise_window \
        -datatype float64 \
        "$denoise_dwi"
    denoise_ap_dwi="${tmp_dir}/${pre}_denoised_ap_dwi.mif"
    mrconvert "$denoise_dwi" -datatype float32 "$denoise_ap_dwi"
    rm -v "$denoise_dwi"
    ap_dwi="$denoise_ap_dwi"
fi

## 2. topup susceptibility distortion correction
pre="20"
topup_tmp_dir="${tmp_dir}/${pre}_topup"
mkdir -p "$topup_tmp_dir"
ap_b0="${tmp_dir}/${pre}_ap_b0.nii.gz"
pa_b0="${tmp_dir}/${pre}_pa_b0.nii.gz"
# Determine b0s to use for topup.
# Extract AP and PA b0 images and subsample to a certain number of b0s.
if [[ "$ap_fmap_b0" != "NULL" ]]; then
    echo "Using AP b=0 images for topup: $ap_fmap_b0"
    mrconvert "$ap_fmap_b0" -axes 0,1,2,-1 -strides 1,2,3,4 "$ap_b0"
else
    dwiextract -bzero "$ap_dwi" "$ap_b0"
fi
if [[ "$pa_fmap_b0" != "NULL" ]]; then
    echo "Using PA b=0 images for topup: $pa_fmap_b0"
    mrconvert "$pa_fmap_b0" -axes 0,1,2,-1 -strides 1,2,3,4 "$pa_b0"
else
    dwiextract -bzero "$pa_dwi" "$pa_b0"
fi

# Create acqparams.txt file.
ap_b0_size=($(mrinfo $ap_b0 -size -quiet))
pa_b0_size=($(mrinfo $pa_b0 -size -quiet))
max_ap_b0="${ap_b0_size[3]}"
max_pa_b0="${pa_b0_size[3]}"
# Limit number of b0s to the user-given amount.
n_ap_b0=$((max_ap_b0 < n_pe_b0s ? max_ap_b0 : n_pe_b0s))
n_pa_b0=$((max_pa_b0 < n_pe_b0s ? max_pa_b0 : n_pe_b0s))
ap_subsample_b0="${tmp_dir}/${pre}_ap_subsample_b0.nii.gz"
pa_subsample_b0="${tmp_dir}/${pre}_pa_subsample_b0.nii.gz"
mrconvert "$ap_b0" -coord 3 0:$((n_ap_b0 - 1)) "$ap_subsample_b0"
mrconvert "$pa_b0" -coord 3 0:$((n_pa_b0 - 1)) "$pa_subsample_b0"
# Create acquisition parameters file.
acqparams="${topup_tmp_dir}/acqparams.txt"
for i in $(seq 1 $n_ap_b0); do
    echo "0 -1 0 $readout_time" >>"$acqparams"
done
for i in $(seq 1 $n_pa_b0); do
    echo "0 1 0 $readout_time" >>"$acqparams"
done
# Combine AP and PA b0s.
b0s="${topup_tmp_dir}/select_b0s.nii.gz"
mrcat "$ap_subsample_b0" "$pa_subsample_b0" -axis 3 "$b0s"
rm "$ap_subsample_b0" "$pa_subsample_b0"

# Run topup (finally)
default_config="${FSLDIR}/etc/flirtsch/b02b0_1.cnf"
pushd "$topup_tmp_dir"
topup \
    --nthr=$N_PROCS $FSL_VERBOSITY \
    --imain="$b0s" \
    --acqp="$acqparams" \
    --out=topup \
    --fout=topup_suscept_field_hz \
    --iout=topup_corrected_b0s \
    --config="$default_config" \
    2>&1 | tee -a topup_stdout.log
popd
topup_output_for_eddy="${topup_tmp_dir}/topup"
undistorted_b0s="${topup_tmp_dir}/topup_corrected_b0s.nii.gz"

## 3. Eddy current + movement-based susceptibility distortion correction
pre="30"
eddy_tmp_dir="${tmp_dir}/${pre}_eddy"
mkdir -p "$eddy_tmp_dir"

# Create a mask for the undistorted b0s.
mean_b0="${tmp_dir}/${pre}_mean_undistorted_b0.nii.gz"
mrmath "$undistorted_b0s" mean -axis 3 "$mean_b0"
bet "$mean_b0" "${eddy_tmp_dir}/undistorted_b0_bet" -f 0.25 -m -n -R
b0_mask="${eddy_tmp_dir}/undistorted_b0_bet_mask.nii.gz"
rm -v "${eddy_tmp_dir}/undistorted_b0_bet.nii.gz"
# Create phase encoding index file.
# The values in the index file correspond to the acqparams.txt file, so copy that file
# to the eddy directory.
cp "$acqparams" "${eddy_tmp_dir}/acqparams.txt"
eddy_index="${eddy_tmp_dir}/index.txt"
if [[ "$ap_dwi" != "NULL" ]]; then
    ap_dwi_size=($(mrinfo $ap_dwi -size -quiet))
    n_ap_dwi="${ap_dwi_size[3]}"
    for i in $(seq 1 $n_ap_dwi); do
        echo -n "1 " >>"$eddy_index"
    done
else
    echo "No AP DWIs provided."
    n_ap_dwi=0
fi
if [[ "$pa_dwi" != "NULL" ]]; then
    pa_dwi_size=($(mrinfo $pa_dwi -size -quiet))
    n_pa_dwi="${pa_dwi_size[3]}"
    pa_idx=$(/usr/bin/cat "${eddy_tmp_dir}/acqparams.txt" | wc -l)
    for i in $(seq 1 $n_pa_dwi); do
        echo -n "$pa_idx " >>"$eddy_index"
    done
else
    echo "No PA DWIs provided."
    n_pa_dwi=0
fi
# Join AP and PA DWIs.
distorted_ap_pa_dwi="${tmp_dir}/${pre}_distorted_ap_pa_dwi.nii.gz"
distorted_ap_pa_bval="${tmp_dir}/${pre}_distorted_ap_pa_dwi.bval"
distorted_ap_pa_bvec="${tmp_dir}/${pre}_distorted_ap_pa_dwi.bvec"
if [[ "$ap_dwi" != "NULL" ]] && [[ "$pa_dwi" != "NULL" ]]; then
    mrcat "$ap_dwi" "$pa_dwi" -axis 3 - |
        mrconvert \
            - \
            -export_grad_fsl "$distorted_ap_pa_bvec" "$distorted_ap_pa_bval" \
            "$distorted_ap_pa_dwi"
elif [[ "$ap_dwi" == "NULL" ]]; then
    mrconvert "$pa_dwi" \
        -export_grad_fsl "$distorted_ap_pa_bvec" "$distorted_ap_pa_bval" \
        "$distorted_ap_pa_dwi"
elif [[ "$pa_dwi" == "NULL" ]]; then
    mrconvert "$ap_dwi" \
        -export_grad_fsl "$distorted_ap_pa_bvec" "$distorted_ap_pa_bval" \
        "$distorted_ap_pa_dwi"
fi

# Run eddy (finally)
pushd "$eddy_tmp_dir"
# Always set the verbose flag so we can capture model parameters that are not otherwise
# saved out.
eddy \
    --data_is_shelled \
    --imain="$distorted_ap_pa_dwi" \
    --mask="$b0_mask" \
    --acqp="${eddy_tmp_dir}/acqparams.txt" \
    --index="$eddy_index" \
    --bvecs="$distorted_ap_pa_bvec" \
    --bvals="$distorted_ap_pa_bval" \
    --topup="$topup_output_for_eddy" \
    --json="$json_sidecar" \
    --niter=7 \
    --nvoxhp=$eddy_num_vox_hp \
    --fwhm=10,8,4,2,0,0,0 \
    --slm=linear \
    --repol \
    --rep_noise \
    --ol_type=both \
    --mporder=16 \
    --estimate_move_by_susceptibility \
    --verbose --no_textout --history --dfields --fields \
    --out=eddy \
    2>&1 | tee -a eddy_stdout.log
popd

# Combine the displacement fields into a single 5D volume.
mrcat "${eddy_tmp_dir}/eddy.eddy_displacement_fields."[0-9]*.nii.gz \
    -axis 4 \
    "${eddy_tmp_dir}/eddy.eddy_displacement_fields.nii.gz"
rm -v "${eddy_tmp_dir}/eddy.eddy_displacement_fields."[0-9]*.nii.gz
eddy_corrected_dwi="${eddy_tmp_dir}/eddy.nii.gz"
eddy_corrected_bvec="${eddy_tmp_dir}/eddy.eddy_rotated_bvecs"
eddy_corrected_dwi_mif="${tmp_dir}/${pre}_eddy_corrected_dwi.mif"
mrconvert \
    "$eddy_corrected_dwi" \
    -fslgrad "$eddy_corrected_bvec" "$distorted_ap_pa_bval" \
    -datatype float32 \
    "$eddy_corrected_dwi_mif"

dwi="$eddy_corrected_dwi_mif"

## 4. B_1 bias field correction
pre="40"
debiased_dwi="${tmp_dir}/${pre}_debiased_dwi.mif"
bias_field="${tmp_dir}/${pre}_bias_field.nii.gz"
dwibiascorrect ants -scratch "$scratch_dir" \
    "$dwi" \
    -bias "$bias_field" \
    "$debiased_dwi"
dwi="$debiased_dwi"

## 5. Final brain masking.
# Combine a BET mask and a registered anatomical mask for the final DWI mask.
pre="50"
median_b0="${tmp_dir}/${pre}_mean_b0.nii.gz"
dwiextract "$dwi" -bzero - | mrmath - median -axis 3 "$median_b0"
bet_mask="${tmp_dir}/${pre}_corrected_b0_bet_mask.nii.gz"
bet "$median_b0" "${tmp_dir}/${pre}_corrected_b0_bet" -f 0.45 -m -n -R
# Register the anatomical image to the DWI space.
antsRegistration --verbose $ANTS_VERBOSITY -d 3 \
    --collapse-output-transforms 1 \
    --interpolation Linear \
    --use-histogram-matching 1 \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --initial-moving-transform [ "$median_b0", "$anat", 1 ] \
    --transform Rigid[ 0.1 ] \
    --metric MI[ "$median_b0", "$anat", 1.0, 32 ] \
    --masks [ "NULL", "$anat_mask"] \
    --convergence [ 1000x500x250x150,1e-6,20 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    --output [ "${tmp_dir}/${pre}_anat_to_median-b0_" ]
ants_anat_to_b0_affine="${tmp_dir}/${pre}_anat_to_median-b0_0GenericAffine.mat"
forward_anat_to_b0_affine_txt="${tmp_dir}/anat_to_median-b0_affine.txt"
"${ANTSPATH}/ConvertTransformFile" 3 \
    "$ants_anat_to_b0_affine" \
    "$forward_anat_to_b0_affine_txt" --homogeneousMatrix --RAS
# Apply transformation to the anat mask.
anat_mask_dwi_space="${tmp_dir}/${pre}_dwi-space_anat_mask.nii.gz"
mrtransform \
    "$anat_mask" \
    -linear "$forward_anat_to_b0_affine_txt" -inverse \
    -template "$median_b0" \
    -interp nearest \
    "$anat_mask_dwi_space"
# Combine the BET mask and the transformed anatomical mask.
dwi_mask="${tmp_dir}/${pre}_final_dwi_mask.nii.gz"
mrcalc "$bet_mask" "$anat_mask_dwi_space" -or "$dwi_mask"
# Fill in any holes in the mask.
"${FSLDIR}/bin/fslmaths" \
    -dt input "$dwi_mask" \
    -fillh \
    "$dwi_mask" -odt input

## 6. Copy files to output directory.
topup_out_dir="${output_dir}/topup"
eddy_out_dir="${output_dir}/eddy"
mkdir -p "$topup_out_dir" "$eddy_out_dir"
cp -v "$topup_tmp_dir"/* "$topup_out_dir"
cp -v "$eddy_tmp_dir"/* "$eddy_out_dir"
# Delete the eddy-corrected full DWI to save space.
rm -v "${eddy_out_dir}/eddy.nii.gz"

# Copy the DWI files to the root of the output directory.
mrconvert \
    "$dwi" \
    -export_grad_fsl "${output_dir}/dwi.bvec" "${output_dir}/dwi.bval" \
    -export_grad_mrtrix "${output_dir}/grad_mrtrix.b" \
    "${output_dir}/dwi.nii.gz"
mrconvert \
    "$dwi_mask" \
    -datatype uint8 \
    "${output_dir}/dwi_mask.nii.gz"
cp -v \
    "$forward_anat_to_b0_affine_txt" \
    "${output_dir}/${anat_label}_to_dwi_fwd-rigid-affine.txt"

if [ "$RM_TMP_DIR" = '1' ]; then
    rm -rvf "$tmp_dir"
else
    echo "Keeping temporary directory $tmp_dir"
fi
