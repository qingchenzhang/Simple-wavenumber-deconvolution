1. This directory contains a simple test for wavenumber-domain deconvolution based on the calculated point spread functions (PSFs), which serves as an implementation of image-domain LSRTM.

2. The input files include the true and migration velocity models, RTM images with and without up-down wavefield decomposition, and the imaging responses of scatter points (PSFs).

3. The main code implements image-domain LSRTM in the wavenumber domain (i.e., PSF deconvolution), where a local window is adopted for the deconvolution process, and the window size is determined based on the maximum velocity.
For further details, please refer to our manuscript Image-domain LSRTM of VSP data with point-spread functions, which has been submitted to Computers & Geosciences.
