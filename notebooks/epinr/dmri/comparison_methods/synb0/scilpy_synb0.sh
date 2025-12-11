#!/usr/bin/bash
set -eou pipefail

# <https://scilpy.readthedocs.io/en/stable/scripts/scil_volume_b0_synthesis.html>
# <https://scilpy.readthedocs.io/en/latest/modules/scilpy.image.html#module-scilpy.image.volume_b0_synthesis>

N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export ITK_NIFTI_SFORM_PERMISSIVE="${ITK_NIFTI_SFORM_PERMISSIVE:-1}"

# Also allow the user to set the "$RM_TMP_DIR" env var to avoid deleting the
# intermediate files for debugging.
RM_TMP_DIR="${RM_TMP_DIR:-1}"
export RM_TMP_DIR
# Create temporary directory, allow user to override with TMP_DIR env var.
tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_synb0_sdc')"}
tmp_dir="$(realpath "$tmp_dir")"

b0="$1"
b0_mask="$2"
acqparams="$3"
pe_dir="$4"
t1w="$5"
t1w_mask="$6"
out_dir="$7"

topup_config="${FSLDIR}/etc/flirtsch/b02b0_1.cnf"

#!IMPORTANT:
# The template used by synb0 for standard space registration is bugged (in version
# 2.2.0, and probably 2.2.1), where the top half of the template is cut off. Also,
# scilpy may look in the wrong location for the template. To fix this, we need to create
# our own fixed template and move it to the correct location. We're not doing that here,
# just make sure you do it.
# The broken template file has this md5sum: 48a79f4f8162b94aa455656d267432ea
# with filename: mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz
####
# # First, find the location where scilpy expects the template to be:
# python -c "import scilpy; import inspect; import pathlib; print(pathlib.Path(inspect.getfile(scilpy)).parent.parent/data"
# # This seems to be one "parent" too far, so it may actually look for:
# "${CONDA_PREFIX}/lib/python3.10/site-packages/data/mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz"
# # When it really should be:
# "${CONDA_PREFIX}/lib/python3.10/site-packages/scilpy/data/mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz"
# # Either way, it's broken, so we copy our corrected template to the location it looks for.
# # Now, to make the corrected template, download the MNI ICBM152 Nonlinear Asymmetric 2009c
# # template from the links below, mask the T1w image using the provided brain mask, and resample
# # to 2.5mm isotropic resolution.
# # <http://www.bic.mni.mcgill.ca/~vfonov/icbm/2009/mni_icbm152_nlin_asym_09c_nifti.zip>
# # <https://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009>
# # Assuming we're in the extracted directory:
# mrcalc mni_icbm152_t1_tal_nlin_asym_09c.nii \
#     mni_icbm152_t1_tal_nlin_asym_09c_mask.nii -mul - |
#     mrgrid - regrid \
#     -template "${CONDA_PREFIX}/lib/python3.10/site-packages/scilpy/data/mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz" \
#     -interp linear \
#     "${CONDA_PREFIX}/lib/python3.10/site-packages/data/mni_icbm152_t1_tal_nlin_asym_09c_masked_2_5.nii.gz"
####
echo "******************************************************"
echo "WARNING: Make sure that the scilpy synb0 template"
echo "has been fixed as per the instructions in the script!"
echo "******************************************************"
#!

# 1. Run synb0 to synthesize the undistorted b0.
mrconvert "$b0" "${tmp_dir}/b0.nii.gz" -datatype float32
mrconvert "$t1w" "${tmp_dir}/t1w.nii.gz" -datatype float32
mrconvert "$t1w_mask" "${tmp_dir}/t1w_mask.nii.gz" -datatype uint8
mrconvert "$b0_mask" "${tmp_dir}/b0_mask.nii.gz" -datatype uint8
scil_volume_b0_synthesis \
    "${tmp_dir}/b0.nii.gz" \
    "${tmp_dir}/b0_mask.nii.gz" \
    "${tmp_dir}/t1w.nii.gz" \
    "${tmp_dir}/t1w_mask.nii.gz" \
    "${tmp_dir}/synb0_undistorted_b0.nii.gz" \
    -v DEBUG

# 2. Run topup with the synthetic b0 having 0 readout time.
# Resample the synthetic b0 to match the input b0 size.
synb0_resampled="${tmp_dir}/synb0_undistorted_b0_resampled.nii.gz"
mrgrid "${tmp_dir}/synb0_undistorted_b0.nii.gz" regrid \
    -template "${tmp_dir}/b0.nii.gz" \
    -interp linear \
    "$synb0_resampled"

# Stack b0 and synthetic b0, create acqparams file for topup.
b0s="${tmp_dir}/input_b0s.nii.gz"
mrcat "${tmp_dir}/b0.nii.gz" "$synb0_resampled" "$b0s"
cp -v "$acqparams" "${tmp_dir}/acqparams.txt"
if [[ "$pe_dir" == "ap" ]]; then
    # Original b0 is AP, synthetic b0 is "PA" with 0 readout time.
    echo "0 1 0 0.000000" >> "${tmp_dir}/acqparams.txt"
elif [[ "$pe_dir" == "pa" ]]; then
    # Original b0 is PA, synthetic b0 is "AP" with 0 readout time.
    echo "0 -1 0 0.000000" >> "${tmp_dir}/acqparams.txt"
else
    echo "Error: pe_dir must be 'ap' or 'pa'. Got: $pe_dir"
    exit 1
fi

pushd "$tmp_dir"
topup \
    --nthr=$N_PROCS --verbose \
    --imain="$b0s" \
    --acqp="${tmp_dir}/acqparams.txt" \
    --out=synb0+topup \
    --fout=suscept_field_hz \
    --iout=synb0+topup_corrected_b0s \
    --config="$topup_config" \
    2>&1 | tee -a synb0+topup_stdout.log
popd

# 3. Save results to output directory.
mkdir --parents "$out_dir"
cp -v \
    "${tmp_dir}/synb0_undistorted_b0.nii.gz" \
    "${tmp_dir}/synb0_undistorted_b0_resampled.nii.gz" \
    "$out_dir"
cp -v \
    "${tmp_dir}/suscept_field_hz.nii.gz" \
    "$tmp_dir"/synb0* \
    "$out_dir"

if [[ "$RM_TMP_DIR" == "1" ]]; then
    rm -rfv "$tmp_dir"
else
    echo "Temporary directory not removed: $tmp_dir"
fi
