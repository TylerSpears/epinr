#!/usr/bin/bash

set -eou pipefail

################################################################################
# Input arguments.
# Help message.
if [ "$#" -lt 3 ]; then
    echo "Usage: $0 [options] <moving_image> <fixed_image> <out_moving_reg_image>"
    echo "Options:"
    echo "  -r                          Resample moving image voxel size to fixed image voxel size."
    echo "  -m <moving_mask>            Moving image mask."
    echo "  -f <fixed_mask>             Fixed image mask."
    echo "  -s <moving_segmentation>    Moving image segmentation."
    echo "  -n <ants_interp>            ANTs interpolation method (default: Linear)."
    echo "  -t <ants_metric>            ANTs metric (CC or MI, default: CC)."
    echo "  -p <ants_metric_params>     ANTs metric parameters (default: depends on metric)."
    echo "  -M <out_moving_mask>        Output registered moving image mask."
    echo "  -S <out_moving_segmentation> Output registered moving image segmentation."
    echo "  -A <out_ants_affine>        Output ANTs affine transform file."
    echo "  -X <out_mrtrix_affine>      Output MRtrix affine transform file."
    echo "  -v                          Verbose mode."
    return 1
fi
# Optional arguments.
resample_mov_spacing=0
ants_metric="CC"
init_mov_tf="1"
ants_metric_params="NULL"
moving_mask="NULL"
moving_segmentation="NULL"
fixed_mask="NULL"
ants_interp="Linear"
out_moving_mask="NULL"
out_moving_segmentation="NULL"
out_ants_affine="NULL"
out_mrtrix_affine="NULL"
VERBOSE="${VERBOSE:-0}"
# Parse optional arguments.
while getopts "rm:f:n:t:p:i:s:M:S:A:X:v" opt; do
    case $opt in
    r)
        resample_mov_spacing=1
        echo "Resampling moving image voxel size to fixed image voxel size: $opt"
        ;;
    m)
        moving_mask="$(realpath "$OPTARG")"
        echo "Moving image mask: $opt, argument: $moving_mask"
        ;;
    f)
        fixed_mask="$(realpath "$OPTARG")"
        echo "Fixed image mask: $opt, argument: $fixed_mask"
        ;;
    s)
        moving_segmentation="$(realpath "$OPTARG")"
        echo "Moving image segmentation: $opt, argument: $moving_segmentation"
        ;;
    n)
        ants_interp="$OPTARG"
        echo "ANTs interpolation method: $opt, argument: $ants_interp"
        ;;
    t)
        # Convert to uppercase and remove whitespace.
        ants_metric="$(printf '%s' "$OPTARG" | tr -d '[:space:]' | tr '[:lower:]' '[:upper:]')"
        echo "ANTs metric: $opt, argument: $ants_metric"
        ;;
    p)
        ants_metric_params="$OPTARG"
        echo "ANTs metric parameters: $opt, argument: $ants_metric_params"
        ;;
    i)
        init_mov_tf="$OPTARG"
        echo "Initial moving transform: $opt, argument: $init_mov_tf"
        ;;
    M)
        out_moving_mask="$(realpath "$OPTARG")"
        echo "Output registered moving image mask: $opt, argument: $out_moving_mask"
        ;;
    S)
        out_moving_segmentation="$(realpath "$OPTARG")"
        echo "Output registered moving image segmentation: $opt, argument: $out_moving_segmentation"
        ;;
    A)
        out_ants_affine="$(realpath "$OPTARG")"
        echo "Output ANTs affine transform file: $opt, argument: $out_ants_affine"
        ;;
    X)
        out_mrtrix_affine="$(realpath "$OPTARG")"
        echo "Output MRtrix affine transform file: $opt, argument: $out_mrtrix_affine"
        ;;
    v)
        VERBOSE='1'
        echo "Verbose mode enabled: $opt"
        ;;
    *)
        echo "Invalid option: $opt"
        return 1
        ;;
    esac
done
# Validate required positional arguments.
if [[ "$out_moving_mask" != "NULL" && "$moving_mask" == "NULL" ]]; then
    echo "Error: Output moving mask specified but no moving mask provided."
    return 1
fi
if [[ "$out_moving_segmentation" != "NULL" && "$moving_segmentation" == "NULL" ]]; then
    echo "Error: Output moving segmentation specified but no moving segmentation provided."
    return 1
fi
# Select default ants metric params if not provided.
if [ "$ants_metric_params" = "NULL" ]; then
    if [ "$ants_metric" = "CC" ]; then
        ants_metric_params="1,3,Regular,0.5,0"
    elif [ "$ants_metric" = "MI" ]; then
        ants_metric_params="1,32,Regular,0.25"
    else
        echo "Error: Unsupported ANTs metric: $ants_metric"
        return 1
    fi
fi
shift $((OPTIND-1))
# Positional arguments, come after optional args.
moving="$1"
fixed="$2"
out_moving_reg="$3"

ANTS_SEED=1491756966
# Default to all processors, if N_PROCS is not set.
N_PROCS=${N_PROCS-"$(nproc)"}
export ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=$N_PROCS
export OMP_NUM_THREADS=$N_PROCS
export MRTRIX_NTHREADS=$N_PROCS
# Control verbosity flags.
# Allow the user to set verbosity to debug the commands in this script.
if [ "$VERBOSE" = '1' ]; then
    MRTRIX_VERBOSITY='-info'
    MRTRIX_LOGLEVEL=2
    ANTS_VERBOSITY=1
else
    MRTRIX_VERBOSITY='-quiet'
    MRTRIX_LOGLEVEL=0
    ANTS_VERBOSITY=0
fi
export MRTRIX_LOGLEVEL
export MRTRIX_VERBOSITY
# Also allow the user to set the "$RM_TMP_DIR" env var to avoid deleting the
# intermediate files for debugging.
RM_TMP_DIR="${RM_TMP_DIR:-1}"
export RM_TMP_DIR
tmp_dir=${TMP_DIR-"$(mktemp -d --suffix='_rigid_reg')"}
tmp_dir="$(realpath "$tmp_dir")"
mkdir --parents "$tmp_dir"
scratch_dir="${tmp_dir}/mrtrix_scratch"
mkdir --parents "$scratch_dir"
export MRTRIX_TMPFILE_DIR="$scratch_dir"
if [ "$VERBOSE" = '1' ]; then
    echo "=========== Temporary Directory $tmp_dir ==================="
fi

# If the user requests saving out the affine in mrtrix format, make sure that the fixed
# and moving images are both in RAS orientation. Otherwise, the transformation
# conversion becomes tricky...
if [ "$out_mrtrix_affine" != "NULL" ]; then
    fixed_strides="$(mrinfo -strides "$fixed")"
    moving_strides="$(mrinfo -strides "$moving")"
    if [[ "$fixed_strides" != "1 2 3" ]] || [[ "$moving_strides" != "1 2 3" ]]; then
        echo "Fixed and/or moving image(s) is/are not in RAS orientation."
        echo "Please convert to RAS orientation when saving affine transform in MRtrix format."
        return 1
    fi
fi

## Perform rigid registration of moving to fixed image.
antsRegistration --verbose $ANTS_VERBOSITY --dimensionality 3 \
    --random-seed $ANTS_SEED \
    --collapse-output-transforms 1 \
    --interpolation Linear \
    --use-histogram-matching 1 \
    --winsorize-image-intensities [ 0.05,0.95 ] \
    --initial-moving-transform [ "$fixed","$moving",$init_mov_tf ] \
    --transform Rigid[ 0.1 ] \
    --metric $ants_metric[ "$fixed","$moving",$ants_metric_params ] \
    --masks [ "$fixed_mask", "$moving_mask"] \
    --convergence [ 1000x500x250x100,1e-6,10 ] \
    --shrink-factors 8x4x2x1 \
    --smoothing-sigmas 3x2x1x0vox \
    --output "${tmp_dir}/moving_reg_fixed_"
ants_affine="${tmp_dir}/moving_reg_fixed_0GenericAffine.mat"

# Apply transformation to the moving image.
# Select output template spacing based on user option.
if [ "$resample_mov_spacing" -eq 1 ]; then
    out_template="$fixed"
else
    # Apply transformation to the moving image, with a target space that keeps the original
    # moving spacing.
    out_template="${tmp_dir}/fixed-grid_with_moving-spacing.nii.gz"
    moving_spacing="$(mrinfo "$moving" -quiet -spacing | tr " " ",")"
    mrgrid "$fixed" regrid -interp nearest -voxel "$moving_spacing" "$out_template"
fi
tmp_out_moving_reg="${tmp_dir}/moving_reg.nii.gz"
antsApplyTransforms --verbose $ANTS_VERBOSITY \
    --dimensionality 3 \
    --input "$moving" \
    --output "$tmp_out_moving_reg" \
    --reference-image "$out_template" \
    --interpolation $ants_interp \
    --transform "$ants_affine" \
    --default-value 0
mrconvert "$tmp_out_moving_reg" "$out_moving_reg" -force

# Apply transformation to the moving mask, if provided.
if [ "$moving_mask" != "NULL" ] && [ "$out_moving_mask" != "NULL" ]; then
    tmp_out_moving_mask="${tmp_dir}/moving_mask_reg.nii.gz"
    antsApplyTransforms --verbose $ANTS_VERBOSITY \
        --dimensionality 3 \
        --input "$moving_mask" \
        --output "$tmp_out_moving_mask" \
        --reference-image "$out_template" \
        --interpolation NearestNeighbor \
        --transform "$ants_affine" \
        --default-value 0
    mrconvert "$tmp_out_moving_mask" "$out_moving_mask" -force
fi
# Apply transformation to the moving segmentation, if provided.
if [ "$out_moving_segmentation" != "NULL" ]; then
    tmp_out_moving_segmentation="${tmp_dir}/moving_segmentation_reg.nii.gz"
    antsApplyTransforms --verbose $ANTS_VERBOSITY \
        --dimensionality 3 \
        --input "$moving_segmentation" \
        --output "$tmp_out_moving_segmentation" \
        --reference-image "$out_template" \
        --interpolation GenericLabel \
        --transform "$ants_affine" \
        --default-value 0
    mrconvert "$tmp_out_moving_segmentation" "$out_moving_segmentation" -force
fi

# Output affine(s) if requested.
if [ "$out_ants_affine" != "NULL" ]; then
    cp --update "$ants_affine" "$out_ants_affine"
fi
if [ "$out_mrtrix_affine" != "NULL" ]; then
    ConvertTransformFile 3 \
        "$ants_affine" \
        "$out_mrtrix_affine" --homogeneousMatrix --RAS
fi

if [ "$RM_TMP_DIR" = '1' ]; then
    rm -rvf "$tmp_dir"
else
    echo "Keeping temporary directory $tmp_dir"
fi
