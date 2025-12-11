#!/usr/bin/bash
set -eou pipefail

# ANTs SyN registration between EPI and anatomical image.
mov_b0="$1"
mov_b0_mask="$2"
fixed_t1w="$3"
fixed_t1w_mask="$4"
output_dir="$5"

mkdir --parents "$output_dir"

N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export ITK_NIFTI_SFORM_PERMISSIVE="${ITK_NIFTI_SFORM_PERMISSIVE:-1}"

# ANTs SyN registration, modified parameters from 'antsIntermodalityIntrasubject.sh'.
# Parameters with a first affine stage.
# --metric MI[ "$d/t1w.nii.gz","$d/b0.nii.gz",1,32,Regular,0.25 ] \
# --transform Affine[ 0.1 ] \
# --restrict-deformation 1x1x1x1x1x1x1x1x1x1x1x1 \
# --convergence [ 1000x500x250x0,1e-7,5 ] \
# --shrink-factors 8x4x2x1 \
# --smoothing-sigmas 4x2x1x0 \
# --use-histogram-matching 1 \
# Found that skipping affine stage and *not* including masks performs better with
# T1w registration.
antsRegistration -d 3 -v 1 --random-seed 1 \
    --float 0 --collapse-output-transforms 1 \
    --initial-moving-transform Identity \
    --interpolation Linear \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    --transform SyN[ 0.1,3,0 ] \
    --restrict-deformation 0x1x0 \
    --metric mattes[ "$fixed_t1w","$mov_b0",1,32 ] \
    --use-histogram-matching 1 \
    --convergence [ 50x50x0,1e-7,5 ] \
    --shrink-factors 4x2x1 \
    --smoothing-sigmas 2x1x0mm \
    --output [ "${output_dir}/b0-reg-t1w_","${output_dir}/b0_warped-to-t1w.nii.gz","${output_dir}/t1w_warped-to-b0.nii.gz" ] |
    tee "${output_dir}/antsRegistration_b0-to-t1w.log"

### antsIntermodalityIntrasubject.sh manual command example
# ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=19 antsRegistration -d 3 -v 1 \
#     -m MI[ /home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/HC_01/dwi-0/t1w.nii.gz,/home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/HC_01/dwi-0/b0.nii.gz,1,32,Regular,0.25 ] \
#     -c [ 1000x500x250x0,1e-7,5 ] \
#     -t Affine[ 0.1 ] \
#     -f 8x4x2x1 \
#     -s 4x2x1x0 \
#     -u 1 \
#     -m mattes[ /home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/HC_01/dwi-0/t1w.nii.gz,/home/tas6hh/mnt/magpie/outputs/vcu_ms_epinr/derivatives/epinr_fmap_learning/HC_01/dwi-0/b0.nii.gz,1,32 ] \
#     -c [ 50x50x0,1e-7,5 ] \
#     -t SyN[ 0.1,3,0 ] \
#     -f 4x2x1 \
#     -s 2x1x0mm \
#     -u 1 \
#     -z 1 \
#     --winsorize-image-intensities [ 0.005, 0.995 ] \
#     -o ants_b0_reg_t1w_
## Full parameter names expanded:
# antsRegistration \
#     --collapse-output-transforms 1 \
#     --dimensionality 3 \
#     --float 1 \
#     --initialize-transforms-per-stage 0 \
#     --interpolation Linear \
#     --output [ ants_susceptibility, ants_susceptibility_Warped.nii.gz ] \
#     --transform SyN[ 0.8, 2.0, 2.0 ] \
#     --metric Mattes[ /out/tmp_workdir/qsiprep_1_0_wf/sub_HC042_ses_01_wf/dwi_preproc_ses_01_wf/hmc_sdc_wf/sdc_wf/syn_sdc_wf/t1_2_ref/masked_brain_trans_inv_trans.nii, /out/tmp_workdir/qsiprep_1_0_wf/sub_HC042_ses_01_wf/dwi_preproc_ses_01_wf/hmc_sdc_wf/extract_b0_series/eddy_corrected_LPS_b0_series_mean.nii.gz, 1, 56 ] \
#     --convergence [ 100x50, 1e-08, 20 ] \
#     --smoothing-sigmas 1.0x0.0vox \
#     --shrink-factors 2x1 \
#     --use-histogram-matching 1 \
#     --restrict-deformation 0.0x1.0x0.0 \
#     --masks [ NULL, NULL ] \
#     --transform SyN[ 0.8, 2.0, 2.0 ] \
#     --metric CC[ /out/tmp_workdir/qsiprep_1_0_wf/sub_HC042_ses_01_wf/dwi_preproc_ses_01_wf/hmc_sdc_wf/sdc_wf/syn_sdc_wf/t1_2_ref/masked_brain_trans_inv_trans.nii, /out/tmp_workdir/qsiprep_1_0_wf/sub_HC042_ses_01_wf/dwi_preproc_ses_01_wf/hmc_sdc_wf/extract_b0_series/eddy_corrected_LPS_b0_series_mean.nii.gz, 1, 5 ] \
#     --convergence [ 20x10, 1e-08, 10 ] \
#     --smoothing-sigmas 1.0x0.0vox \
#     --shrink-factors 1x1 \
#     --use-histogram-matching 1 \
#     --restrict-deformation 0.0x1.0x0.0 \
#     --masks [ /out/tmp_workdir/qsiprep_1_0_wf/sub_HC042_ses_01_wf/dwi_preproc_ses_01_wf/hmc_sdc_wf/sdc_wf/syn_sdc_wf/threshold_atlas/mni_lps_fmap_atlas_trans_threshbin.nii.gz, NULL ] \
#     --winsorize-image-intensities [ 0.001, 1.0 ]  \
#     --write-composite-transform 0
