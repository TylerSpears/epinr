# Notes for MICA-MICS Dataset Preprocessing

* DWIs were processed with the 'preproc_dwi.sh' script.
* All preprocessing happened in RAS orientation.
* There were 4 b0 images extracted from the dwi data:
    * 1 AP b0 from the start of the 'acq-b300-11_dir-AP' volume
    * 1 AP b0 from the start of the 'acq-b700-11_dir-AP' volume
    * 2 PA b0s from the start of the 'dir-PA_dwi' volume
* All DWIs, including the b0s, were denoised using MRtrix's 'dwidenoise' command, with a denoise extent window of 5. This method uses information from all DWIs for the Marchenko-Pastur PCA estimation, so technically there are b0s present in the denoising process that should not be available at a true "test time," but that only effects the denoising of voxel intensities, with no regard to structure or distortion.
