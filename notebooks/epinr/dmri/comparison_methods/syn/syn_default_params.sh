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

### ANTs SyN registration, default parameters from the 'antsRegistrationSyN.sh' script.
# Not optimized for inter-modal registration, except for restricting deformation to phase-encoding direction.
antsRegistration --verbose 1 --random-seed 1 --dimensionality 3 \
    --float 0 --collapse-output-transforms 1 \
    --interpolation Linear \
    --use-histogram-matching 1 \
    --winsorize-image-intensities [ 0.005,0.995 ] \
    -x [ "$fixed_t1w_mask","$mov_b0_mask" ] \
    --initial-moving-transform Identity \
    --transform SyN[ 0.2,3,0 ] \
    --metric CC[ "$fixed_t1w","$mov_b0",1,4 ] \
    --restrict-deformation 0.0x1.0x0.0 \
    --convergence [ 100x70x50x20,1e-6,10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    --output [ "${output_dir}/b0-reg-t1w_","${output_dir}/b0_warped-to-t1w.nii.gz","${output_dir}/t1w_warped-to-b0.nii.gz" ] |
    tee "${output_dir}/antsRegistration_b0-to-t1w.log"


## ANTs SyN registration, modified parameters from 'antsIntermodalityIntrasubject.sh'.
# antsRegistration -d 3 -v 1 --random-seed 1 \
#     --float 0 --collapse-output-transforms 1 \
#     --initial-moving-transform Identity \
#     --interpolation Linear \
#     --winsorize-image-intensities [ 0.005,0.995 ] \
#     --transform SyN[ 0.1,3,0 ] \
#     --restrict-deformation 0x1x0 \
#     --metric mattes[ "$d/t1w.nii.gz","$d/b0.nii.gz",1,32 ] \
#     --use-histogram-matching 1 \
#     --convergence [ 50x50x0,1e-7,5 ] \
#     --shrink-factors 4x2x1 \
#     --smoothing-sigmas 2x1x0mm \
#     --output [ "${out_dir}/b0-reg-t1w_","${out_dir}/b0_warped-to-t1w.nii.gz","${out_dir}/t1w_warped-to-b0.nii.gz" ] |
#     tee "${out_dir}/antsRegistration_b0-to-t1w.log"

# Found that skipping affine stage and *not* including masks performs better with
# T1w registration. Copied here for reference.
# --metric MI[ "$d/t1w.nii.gz","$d/b0.nii.gz",1,32,Regular,0.25 ] \
# --transform Affine[ 0.1 ] \
# --restrict-deformation 1x1x1x1x1x1x1x1x1x1x1x1 \
# --convergence [ 1000x500x250x0,1e-7,5 ] \
# --shrink-factors 8x4x2x1 \
# --smoothing-sigmas 4x2x1x0 \
# --use-histogram-matching 1 \
