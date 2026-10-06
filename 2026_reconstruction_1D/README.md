# PFE Results
See below for how to reproduce.
Noise Calibration of v2 Single Pixel Camera, where acquisition is modelled as: 

$$
\mathbf{Y} \sim \gamma\,\mathcal{P}(\mathbf{A}\mathbf{F}) + \mathcal{N}(\mu_d, \sigma_d^2)
$$

Where $\mathbf{Y}$ are the raw measurements, $\gamma$, $\mu_d$ and $\sigma_d^2$ are estimated experimentally, $\mathbf{A}$ is the known acquisition matrix (split Hadamard), and $\mathbf{F}$ is the hyperspectral representation of the scene. 
- $\mathcal{P}$: Poisson distribution (shot noise),
- $\mathcal{N}$: Gaussian distribution (electronic/read noise, assumed additive and independent of the signal),
- $\gamma$: overall system gain (counts per electron),
- $\mu_d$: detector offset (bias level),
- $\sigma_d^2$: read-noise variance.

The measurement matrix can be split into positive and negative components:

$$
\mathbf{Y}^+ \sim \gamma\mathcal{P}(\mathbf{H}^+\mathbf{F}) + \mathcal{N}(\mu_d, \sigma_d^2)
$$

$$
\mathbf{Y}^- \sim \gamma\mathcal{P}(\mathbf{H}^-\mathbf{F}) + \mathcal{N}(\mu_d, \sigma_d^2)
$$


$$
\mathbf{Y}^+ - \mathbf{Y}^- \sim \gamma\text{Skellam}(\mathbf{A}^+\mathbf{F}, \mathbf{A}^-\mathbf{F}) + \mathcal{N}(0, 2\sigma_d^2)
$$

Subtracting the measurement pairs removes the fixed detector offset: the difference of two independent $\mathcal{N}(\mu_d, \sigma_d^2)$ variables is $\mathcal{N}(0, 2\sigma_d^2)$ — the dark offsets $\mu_d$ cancel and the variances sum. The counting part is Skellam-distributed (the difference of two independent Poisson variables). For the underlying counting process itself — i.e. before the gain $\gamma$ and read-noise term are reintroduced — the mean and variance are

$$
\mathbb{E}[\mathbf{A}^+\mathbf{F} - \mathbf{A}^-\mathbf{F}] = \mathbf{H}\mathbf{F}
$$

$$
\text{Var}(\mathbf{A}^+\mathbf{F} - \mathbf{A}^-\mathbf{F}) = \mathbf{1}_{N_x}^\top \mathbf{F}
$$

As such, the mean of the differenced measurement recovers the ideal virtual-Hadamard signal, while its noise variance is set by the total (unmodulated) photon flux $\mathbf{1}_{N_x}.$

## Reconstruction

Implementation of Direct reconstruction: 

$$\mathbf{F}_{pinv} = \mathbf{H}^\dagger(\frac{\mathbf{Y}^+-\mathbf{Y}^-}{\gamma})$$

Tikhonov Regularisation 

$$
$$

and Neural Network Denoising on simulated and experimental data.

Example:






 Noise Estimation

Dark noise images and constants were estimated from a set of K dark acquisitions:

$$
\hat{\mu}_d(x,y) = \frac{1}{K}\sum_{k=1}^K D_k(x,y) \approx \mathbb{E}[\mathbf{D}]
$$

$$
\hat{\sigma}_d^2(x,y) = \frac{1}{K-1}\sum_{k=1}^K \big(D_k(x,y) - \hat{\mu}_d(x,y)\big)^2 \approx \text{Var}[\mathbf{D}]
$$

Estimated dark offset and noise parameters for various acquisition configurations of the v2 single pixel camera:

**18.06dB**

| | $\bar\mu_d$ | $\bar\sigma_d^2$ | $\bar\sigma_d$ |
|---|---|---|---|
| 1.0ms | 707.21 ± 28.25 | 46.42 ± 12.33 | 6.76 ± 0.88 |
| 10.0ms | 717.94 ± 27.91 | 48.15 ± 12.89 | 6.88 ± 0.90 |
| 100.0ms | 804.88 ± 30.38 | 59.37 ± 18.40 | 7.62 ± 1.17 |
| 1000.0ms | 2054.86 ± 146.29 | 454.38 ± 63.26 | 21.26 ± 1.51 |

**12.04dB**

| | $\bar\mu_d$ | $\bar\sigma_d^2$ | $\bar\sigma_d$ |
|---|---|---|---|
| 1.0ms | 243.42 ± 11.94 | 24.30 ± 23.20 | 4.38 ± 2.27 |
| 10.0ms | 247.03 ± 11.02 | 21.15 ± 21.80 | 4.04 ± 2.19 |
| 100.0ms | 270.86 ± 12.05 | 24.41 ± 20.63 | 4.53 ± 1.98 |

**6.02dB**

| | $\bar\mu_d$ | $\bar\sigma_d^2$ | $\bar\sigma_d$ |
|---|---|---|---|
| 1.0ms | 178.21 ± 8.37 | 11.05 ± 12.09 | 3.06 ± 1.30 |
| 10.0ms | 181.01 ± 8.18 | 10.70 ± 12.55 | 3.00 ± 1.31 |
| 100.0ms | 207.62 ± 12.01 | 30.64 ± 19.32 | 5.23 ± 1.82 |

**0.0dB**

| | $\bar\mu_d$ | $\bar\sigma_d^2$ | $\bar\sigma_d$ |
|---|---|---|---|
| 1.0ms | 91.92 ± 5.53 | 15.54 ± 6.32 | 3.85 ± 0.85 |
| 10.0ms | 96.51 ± 4.86 | 12.44 ± 6.72 | 3.38 ± 1.02 |
| 100.0ms | 124.17 ± 2.58 | 2.23 ± 2.56 | 1.32 ± 0.71 |


## Gain estimation
2 methods to estimate count/photon conversion gain of camera at various configurations, with the following gain constants estimated:

| Configuration | γ FFP (counts/photon) | γ TE (counts/photon) |
|---|---|---|
| 18.06dB | 0.4427 | 0.316 |
| 12.04dB | 0.135 | 0.115 |
| 6.02dB | 0.173 | 0.190 |
| 0dB | 0.181 | 0.117 |

example gain map and gain constant in region of interest for camera gain 18.06dB:

<img width="394" height="198" alt="image" src="https://github.com/user-attachments/assets/0893d04b-7b1d-44f8-9864-35700ce2eab0" />

<img width="551" height="414" alt="image" src="https://github.com/user-attachments/assets/ae57e8a1-6651-4d48-818c-91049e3ad0c8" />

## Downloading raw data
Run `download_data.py`to download raw v2 data from PILOT.
> **Note:** the raw data for some versions (eg lisaCat) are not stored in dictionaries and so the folder had to be downloaded by hand and each file read into a single array, then formatted. See second half of script.

## Formating Raw Data
Run `format_data.py`to format the raw data downloaded using `download_data.py`. This script reorders the raw measurements into chronological order (measures may not have been acquired in this order - acquisition order information retrieved from the relevant metadata file), checks dimensions, and bins according to bin_fact. 

Adjust formatting parameters according to raw data in question if necessary. The final measurement array is saved as `m_binned.npy`, and is of dimensions $(N_y \times \Lambda \times P)$, where $N_y$ is the spatial dimension, $\Lambda$ is the spectral dimension, and $P$ is the number of nonegative patterns applied (twice the number of virtual rows $K$ in the case of negative virtual matrices).

## Dark Noise
Dark noise images and constants were estimated from a set of K dark acquisitions:

$$
\hat{\mu}_d(x,y) = \frac{1}{K}\sum_{k=1}^K D_k(x,y) \approx \mathbb{E}[\mathbf{D}]
$$

$$
\hat{\sigma}_d^2(x,y) = \frac{1}{K-1}\sum_{k=1}^K \big(D_k(x,y) - \hat{\mu}_d(x,y)\big)^2 \approx \text{Var}[\mathbf{D}]
$$
> **Note:** the raw dark data must already be stored on your local machine. The dark acquisitions used in this project were saved locally and never uploaded to the PILOT.


To reproduce the calculation of the dark noise images and constants for a given acquisition configuration, run `dark_estimation.py`.

To loop over a set of acquisition parameters, run `dark_estimation_loop.py`, adjusting it to the desired parameters.

In future the data may be uploaded to the PILOT, in which case the `download_data.py` script can be used.

## Gain Conversion

Gain conversion images (gain per pixel) and gain constants (average over pixels in gain image) were calculated from a set of uniformly illuminated acquisitions using the photon transfer curve.

To calculate the gain images for a given camera configuration, run `conversion_gain_estim.py`.

> **Note:** once again, the raw bright data must already be stored on your local machine.

Two methods for calculating a constant gain parameter per camera configuration are implemented compared:

**Flatfield Pair Method:** Expected value and variation of count rate calculated from Photon Transfer Curve using spatial averaging over pixels in a region of interest $->$ technically only two acquisitions are needed, but here I also averaged temporally (over 500 pairs) because I had 1000 acquisitions per intensity level. 

Acquire a flat-field image pair $(Y_1, Y_2)$ under identical illumination and take the difference $D = Y_1 - Y_2$. Since $Y_1, Y_2$ are independent draws of the same random variable $Y$:

$$
\mathbb{E}[D] = 0
$$

$$
\text{Var}[D] = 2\,\text{Var}[Y] = 2(\gamma^2 s + \sigma_d^2)
$$

Differencing cancels any fixed pattern offset but doubles the variance, so

$$
\hat{\sigma}^2 := \frac{\text{Var}[D]}{2} = \gamma^2 s + \sigma_d^2
$$

Let $\hat{S}$ denote the averaged, dark-subtracted mean of $Y_1, Y_2$ — each containing $N$ pixels in total:

$$
\bar{Y}_1 = \frac{1}{N}\sum_{n=1}^N \big(Y_1(n) - \mu_d\big)
$$

$$
\bar{Y}_2 = \frac{1}{N}\sum_{n=1}^N \big(Y_2(n) - \mu_d\big)
$$

$$
\hat{S} = \frac{\bar{Y}_1 + \bar{Y}_2}{2}
$$

From above, the dark-subtracted signal is $\gamma s$, so $\hat{S} \approx \gamma s$. Substituting $s = \hat{S}/\gamma$ gives

$$
\hat{\sigma}^2 = \gamma^2\left(\frac{\hat{S}}{\gamma}\right) + \sigma_d^2 = \gamma\hat{S} + \sigma_d^2
$$

This is a straight line in the directly measurable, dark-subtracted quantity $\hat{S}$: the slope recovers the gain $\gamma$, and the intercept is $\sigma_d^2$.


**Temporal Estimate Method:** Gain value calculated per pixel from expected value and variation images, which are the temporal average over 1000 aqcuisitions. The constant gain is then the spatial average within a region of interest. 
At each illumination level, $N$ repeated exposures are acquired, $\mathbf{Y} = \{Y_1, \dots, Y_N\}$, and the temporal sample mean and variance per pixel are computed as:

$$
\hat{\mu}(x,y) = \frac{1}{N}\sum_{n=1}^N Y_n(x,y), \qquad \hat{\sigma}^2(x,y) = \frac{1}{N-1}\sum_{n=1}^N \big(Y_n(x,y) - \hat{\mu}(x,y)\big)^2
$$

Plotting $(\hat{\mu}, \hat{\sigma}^2)$ pairs and fitting a line gives the gain from the slope. This can be done per pixel, for a full gain map, or using the spatial average of the mean and variance images, for a single scalar gain.

Both methods are implemented in the python script, with adjustable parameters to take into account the various acquisition and reconstruction parameters (i.e; camera gain, binning etc)

## Image covariance prior

To calculate the 1D covariance prior over the rows of a set of natural images (ImageNet dataset) run `covariance_prior.py`with the desired parameters (image size, normalisation etc). The resulting covariance matrix is saved to a data folder. 

## Reconstruction
Ruņ `reconstruction.py` to reconstruct data that has been downloaded from PILOT using `download_data.py`and formatted using `format_data.py`. 

Pseudoinverse and tikhonov reconstruction is implemented, with the option to apply denoising using a UNet() trained by running `train.py`and the desired parameters. 

The reconstructed cubes are plotted as the sum over wavelength channels and per wavelength channel individually. 


## Training Denoiser
To train a UNet() denoiser, run `train.py` with the desired parameters. Trained weights will be saved to a 'model' folder and can then be used during reconstruction. 
