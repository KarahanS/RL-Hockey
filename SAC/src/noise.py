import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__)))  # Adds current directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  # Adds parent directory
from abc import ABC, abstractmethod
import numpy as np
from typing import Optional
from numpy.typing import DTypeLike
from utils import powerlaw_psd_gaussian

# FYI: We do not need to implement ColoredNoiseDist or PinkNoiseDist as they simply use Noise and tanh squashing
# which is already done by our Actor
# Source: https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/noise.py
#         https://github.com/martius-lab/pink-noise-rl/tree/main/pink


class ActionNoise(ABC):
    """The action noise base class"""

    def __init__(self) -> None:
        super().__init__()

    def reset(self) -> None:
        """Call end of episode reset for the noise"""
        pass

    @abstractmethod
    def __call__(self) -> np.ndarray:
        raise NotImplementedError()


class NormalActionNoise(ActionNoise):
    """
    A Gaussian action noise.
    :param mean: Mean value of the noise
    :param sigma: Scale of the noise (std here)
    :param dtype: Type of the output noise
    """

    def __init__(
        self, mean: np.ndarray, sigma: np.ndarray, dtype: DTypeLike = np.float32
    ) -> None:
        self._mu = mean
        self._sigma = sigma
        self._dtype = dtype
        super().__init__()

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma).astype(self._dtype)

    def __repr__(self) -> str:
        return f"NormalActionNoise(mu={self._mu}, sigma={self._sigma})"

    def get_state(self):
        """Return the current internal state."""
        return {}

    def set_state(self, state):
        """Restore the internal state. If missing, reset to default."""
        pass


class OrnsteinUhlenbeckActionNoise(ActionNoise):
    """
    An Ornstein Uhlenbeck action noise, this is designed to approximate Brownian motion with friction.
    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    :param mean: Mean of the noise
    :param sigma: Scale of the noise
    :param theta: Rate of mean reversion
    :param dt: Timestep for the noise
    :param initial_noise: Initial value for the noise output, (if None: 0)
    :param dtype: Type of the output noise
    """

    def __init__(
        self,
        mean: np.ndarray,
        sigma: np.ndarray,
        theta: float = 0.15,
        dt: float = 1e-2,
        initial_noise: Optional[np.ndarray] = None,
        dtype: DTypeLike = np.float32,
    ) -> None:
        self._theta = theta
        self._mu = mean
        self._sigma = sigma
        self._dt = dt
        self._dtype = dtype
        self.initial_noise = initial_noise
        self.noise_prev = np.zeros_like(self._mu)
        self.sqrtdt = np.sqrt(self._dt)
        self.reset()
        super().__init__()

    def __call__(self) -> np.ndarray:
        noise = (
            self.noise_prev
            + self._theta * (self._mu - self.noise_prev) * self._dt
            + self._sigma * self.sqrtdt * np.random.normal(size=self._mu.shape)
        )
        self.noise_prev = noise
        return noise.astype(self._dtype)

    def reset(self) -> None:
        """Reset the Ornstein Uhlenbeck noise, to the initial position"""
        self.noise_prev = (
            self.initial_noise
            if self.initial_noise is not None
            else np.zeros_like(self._mu)
        )

    def __repr__(self) -> str:
        return f"OrnsteinUhlenbeckActionNoise(mu={self._mu}, sigma={self._sigma})"

    def get_state(self):
            """Return the current internal state."""
            return {"noise_prev": self.noise_prev.copy()}

    def set_state(self, state):
        """Restore the internal state. If missing, reset to default."""
        try:
            self.noise_prev = state["noise_prev"].copy()
        except KeyError:
            print("Warning: OrnsteinUhlenbeckActionNoise state missing. Resetting noise.")
            self.reset()

class ColoredActionNoise(ActionNoise):
    def __init__(self, beta, sigma, seq_len, action_dim=None, rng=None):
        """Action noise from a colored noise process.

        Parameters
        ----------
        beta : float or array_like
            Exponent(s) of colored noise power-law spectra. If it is a single float, then `action_dim` has to be
            specified and the noise will be sampled in a vectorized manner for each action dimension. If it is
            array_like, then it specifies one beta for each action dimension. This allows different betas for different
            action dimensions, but sampling might be slower for high-dimensional action spaces.
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled colored noise signals. If sampled for longer than `seq_len` steps, a new
            colored noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int, optional
            Dimensionality of the action space. If passed, `beta` has to be a single float and the noise will be
            sampled in a vectorized manner for each action dimension.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__()
        assert (action_dim is not None) == np.isscalar(
            beta
        ), "`action_dim` has to be specified if and only if `beta` is a scalar."

        self.sigma = (
            np.full(action_dim or len(beta), sigma)
            if np.isscalar(sigma)
            else np.asarray(sigma)
        )

        if np.isscalar(beta):
            self.beta = beta
            self.gen = ColoredNoiseProcess(
                beta=self.beta, scale=self.sigma, size=(action_dim, seq_len), rng=rng
            )
        else:
            self.beta = np.asarray(beta)
            self.gen = [
                ColoredNoiseProcess(beta=b, scale=s, size=seq_len, rng=rng)
                for b, s in zip(self.beta, self.sigma)
            ]

    def __call__(self) -> np.ndarray:
        return (
            self.gen.sample()
            if np.isscalar(self.beta)
            else np.asarray([g.sample() for g in self.gen])
        )

    def __repr__(self) -> str:
        return f"ColoredActionNoise(beta={self.beta}, sigma={self.sigma})"
    
    def get_state(self):
            """Return the internal state of the underlying noise process(es)."""
            if np.isscalar(self.beta):
                return self.gen.get_state()
            else:
                # When beta is not scalar, gen is a list of processes.
                return [g.get_state() for g in self.gen]

    def set_state(self, state):
        """Restore the internal state of the underlying noise process(es)."""
        if np.isscalar(self.beta):
            self.gen.set_state(state)
        else:
            if isinstance(state, list) and len(state) == len(self.gen):
                for g, s in zip(self.gen, state):
                    g.set_state(s)
            else:
                print("Warning: ColoredActionNoise state format unexpected. Resetting noise.")


class PinkActionNoise(ColoredActionNoise):
    def __init__(self, sigma, seq_len, action_dim, rng=None):
        """Action noise from a pink noise process.

        Parameters
        ----------
        sigma : float or array_like
            Noise scale(s) of colored noise signals. Either a single float to be used for all action dimensions, or
            an array_like of the same dimensionality as the action space (one scale for each action dimension).
        seq_len : int
            Length of sampled pink noise signals. If sampled for longer than `seq_len` steps, a new
            pink noise signal of the same length is sampled. Should usually be set to the episode length
            (horizon) of the RL task.
        action_dim : int
            Dimensionality of the action space.
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        super().__init__(1, sigma, seq_len, action_dim, rng)


class ColoredNoiseProcess:
    """Infinite colored noise process.

    Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences the
    PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

    Methods
    -------
    sample(T=1)
        Sample `T` timesteps from the colored noise process.
    reset()
        Reset the buffer with a new time series.
    """

    def __init__(self, beta, size, scale=1, max_period=None, rng=None):
        """Infinite colored noise process.

        Implemented as a buffer: every `size[-1]` samples, a cut to a new time series starts. As this cut influences
        the PSD of the combined signal, the maximum period (1 / low-frequency cutoff) can be specified.

        Parameters
        ----------
        beta : float
            Exponent of colored noise power-law spectrum.
        size : int or tuple of int
            Shape of the sampled colored noise signals. The last dimension (`size[-1]`) specifies the time range, and
            is thus ths maximum possible correlation length of the combined signal.
        scale : int, optional, by default 1
            Scale parameter with which samples are multiplied
        max_period : float, optional, by default None
            Maximum correlation length of sampled colored noise singals (1 / low-frequency cutoff). If None, it is
            automatically set to `size[-1]` (the sequence length).
        rng : np.random.Generator, optional
            Random number generator (for reproducibility). If not passed, a new random number generator is created by
            calling `np.random.default_rng()`.
        """
        self.beta = beta
        if max_period is None:
            self.minimum_frequency = 0
        else:
            self.minimum_frequency = 1 / max_period
        self.scale = scale
        self.rng = rng

        # The last component of size is the time index
        try:
            self.size = list(size)
        except TypeError:
            self.size = [size]
        self.time_steps = self.size[-1]

        # Fill buffer and reset index
        self.reset()

    def reset(self):
        """Reset the buffer with a new time series."""
        self.buffer = powerlaw_psd_gaussian(
            exponent=self.beta,
            size=self.size,
            fmin=self.minimum_frequency,
            rng=self.rng,
        )
        self.idx = 0

    def sample(self, T=1):
        """
        Sample `T` timesteps from the colored noise process.

        The buffer is automatically refilled when necessary.

        Parameters
        ----------
        T : int, optional, by default 1
            Number of samples to draw

        Returns
        -------
        array_like
            Sampled vector of shape `(*size[:-1], T)`
        """
        n = 0
        ret = []
        while n < T:
            if self.idx >= self.time_steps:
                self.reset()
            m = min(T - n, self.time_steps - self.idx)
            ret.append(self.buffer[..., self.idx : (self.idx + m)])
            n += m
            self.idx += m

        ret = self.scale * np.concatenate(ret, axis=-1)
        return ret if n > 1 else ret[..., 0]
    
    def get_state(self):
        """Return the current buffer and index state."""
        return {"buffer": self.buffer.copy(), "idx": self.idx}

    def set_state(self, state):
        """Restore the internal state. If missing, reset."""
        try:
            self.buffer = state["buffer"].copy()
            self.idx = state["idx"]
        except KeyError:
            print("Warning: ColoredNoiseProcess state missing. Resetting noise process.")
            self.reset()