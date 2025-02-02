import numpy as np
from numpy.fft import irfft, rfftfreq

def powerlaw_psd_gaussian(exponent, size, fmin=0, rng=None):
    """Gaussian (1/f)**beta noise.

    Based on the algorithm in:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995)

    Normalised to unit variance

    Parameters:
    -----------

    exponent : float
        The power-spectrum of the generated noise is proportional to

        S(f) = (1 / f)**beta
        flicker / pink noise:   exponent beta = 1
        brown noise:            exponent beta = 2

        Furthermore, the autocorrelation decays proportional to lag**-gamma
        with gamma = 1 - beta for 0 < beta < 1.
        There may be finite-size issues for beta close to one.

    shape : int or iterable
        The output has the given shape, and the desired power spectrum in
        the last coordinate. That is, the last dimension is taken as time,
        and all other components are independent.

    fmin : float, optional
        Low-frequency cutoff.
        Default: 0 corresponds to original paper.

        The power-spectrum below fmin is flat. fmin is defined relative
        to a unit sampling rate (see numpy's rfftfreq). For convenience,
        the passed value is mapped to max(fmin, 1/samples) internally
        since 1/samples is the lowest possible finite frequency in the
        sample. The largest possible value is fmin = 0.5, the Nyquist
        frequency. The output for this value is white noise.

    rng : np.random.Generator, optional
        Random number generator (for reproducibility). If not passed, a new
        random number generator is created by calling
        `np.random.default_rng()`.


    Returns
    -------
    out : array
        The samples.

    Examples:
    ---------

    >>> # generate 1/f noise == pink noise == flicker noise
    >>> import colorednoise as cn
    >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """

    # Make sure size is a list so we can iterate it and assign to it.
    try:
        size = list(size)
    except TypeError:
        size = [size]

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1./samples)    # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = np.sum(s_scale < fmin)   # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale**(-exponent/2.)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.    # correct f = +-0.5
    sigma = 2 * np.sqrt(np.sum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(None,) * dims_to_add + (Ellipsis,)]

    # Generate scaled random power + phase
    if rng is None:
        rng = np.random.default_rng()
    sr = rng.normal(scale=s_scale, size=size)
    si = rng.normal(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= np.sqrt(2)    # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= np.sqrt(2)    # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1J * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


class PinkNoise:
    def __init__(self, action_dim, max_steps, exponent=1.0, fmin=0.0, rng=None):
        """
        Initializes the PinkNoise generator.

        Parameters:
        -----------
        action_dim : int
            The dimensionality of the action space.
        max_steps : int
            The maximum number of steps per episode.
        exponent : float, optional
            The exponent for the power-law noise. Default is 1.0 for pink noise.
        fmin : float, optional
            The low-frequency cutoff. Default is 0.0.
        rng : np.random.Generator, optional
            Random number generator for reproducibility.
        """
        self.action_dim = action_dim
        self.exponent = exponent
        self.fmin = fmin
        self.rng = rng if rng else np.random.default_rng()
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        """
        Resets the noise sequence for a new episode using the predefined max_steps.
        """
        self.noise_sequence = powerlaw_psd_gaussian(
            exponent=self.exponent,
            size=(self.action_dim, self.max_steps),
            fmin=self.fmin,
            rng=self.rng
        )
        self.current_step = 0

    def get_noise(self):
        """
        Retrieves the next noise sample.

        Returns:
        --------
        np.ndarray
            The noise to be added to the action.
        """
        if self.current_step >= self.noise_sequence.shape[1]:
            # If we've exhausted the current sequence, generate a new one
            self.reset()
        noise = self.noise_sequence[:, self.current_step]
        self.current_step += 1
        return noise
