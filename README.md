**Question 1** - **Imaging Science**

I implemented the Retinex-based reflectance recovery model to separate illumination and reflectance from a single image captured under uneven lighting. The goal was to recover the true surface appearance by removing lighting variations while preserving texture details. I converted the input image into the logarithmic domain, where the relationship between illumination and reflectance becomes additive. Using a manually created Gaussian kernel and separable convolution, I estimated the smooth illumination component and isolated the detailed reflectance component. The recovered reflectance was then exponentiated and normalized to obtain an image with uniform brightness. For color images, the algorithm was extended to jointly correct illumination across all RGB channels while maintaining accurate color ratios. The final outputs, including illumination maps, reflectance maps, and histograms, were generated and saved automatically to demonstrate successful lighting correction and texture recovery.

**Question 2** - **Computer Vision**

This project estimates the camera intrinsic parameters and radial distortion coefficients from a single photograph of a checkerboard (rectangular grid) captured using an unknown camera. The input image may include perspective distortion, lighting variations, or mild noise.

The workflow involves detecting the grid corners, formulating a robust cost function that minimizes the reprojection error between the detected points and model-predicted ones, and using non-linear least squares optimization (Huber loss) for stability. A RANSAC-based refinement step ensures that outliers are rejected, improving the robustness of the parameter estimation.

Finally, the estimated camera intrinsics and distortion coefficients are used to undistort the image, producing a geometrically corrected version where straight lines appear straight again. The results include the detected corner visualization, undistorted image, and computed reprojection error, all saved in the output folder for analysis.

**Question 3** - **Transliteration**
   
`Training Details`
The model was trained on a representative subset of the complete dataset to ensure computational stability and avoid runtime interruptions during training. This setup was designed to demonstrate the model’s architecture and performance within the available compute limits.
 - Number of Epochs: Training was intentionally limited to a minimal number of epochs (≤30) due to GPU and time constraints. While longer training would likely improve accuracy, the focus here was to present a stable and efficient implementation for evaluation purposes.
 - Early Stopping: An early stopping mechanism was implemented with `restore_best_weights=True` to automatically retain the most optimal model weights based on validation performance.
 - Model Saving Strategy: After each epoch, only the best-performing model (lowest validation loss) is saved as `conditioned_seq2seq_models/best_conditioned_seq2seq.h5` Older checkpoints are automatically removed to maintain a clean directory and conserve storage space.

**Note:** Although I initially intended to train the model for more epochs to achieve higher accuracy, GPU runtime limitations restricted extended training.
