import typing

import numpy as np
import scipy.optimize as spo
from scipy.special import erf

from smac.epm.base_epm import AbstractEPM


# -------------------------------------------------- Kernels ----------------------------------------------------------
class BaseKernel(object):
    def __init__(self, data=None, bandwidth=None, fix_boundary=False, num_values=None):

        self.data = data
        self.bw = bandwidth
        self.fix_boundary = fix_boundary

        if num_values is None:
            num_values = len(np.unique(data))
        self.num_values = num_values

        if data is not None:
            self.weights = self._compute_weights()

    def set_bandwidth(self, bandwidth):
        self.bw = bandwidth
        self.weights = self._compute_weights()

    def _compute_weights(self):
        return tuple(1.)

    def __call__(self, x_test):
        raise NotImplementedError

    def sample(self, sample_indices=None, num_samples=1):
        raise NotImplementedError


class Gaussian(BaseKernel):

    def _compute_weights(self):
        if not self.fix_boundary:
            return tuple(1.)

        weights = np.zeros(self.data.shape[0])
        for i, d in enumerate(self.data):
            weights[i] = 2. / (erf((1 - d) / (np.sqrt(2) * self.bw)) + erf(d / (np.sqrt(2) * self.bw)))

        return weights[:, None]

    def __call__(self, x_test):
        distances = x_test[None, :] - self.data[:, None]
        pdfs = np.exp(-0.5 * np.power(distances / self.bw, 2)) / (2.5066282746310002 * self.bw)

        # reweigh to compensate for boundaries
        pdfs *= self.weights

        return pdfs

    def sample(self, sample_indices=None, num_samples=1):
        """ returns samples according to the KDE

            Parameters
            ----------
                sample_indices: list of ints
                    Indices into the training data used as centers for the samples

                num_samples: int
                    if samples_indices is None, this specifies how many samples
                    are drawn.

        """
        if sample_indices is None:
            sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
        samples = self.data[sample_indices]

        delta = np.random.normal(size=num_samples) * self.bw
        samples += delta
        oob_idx = np.argwhere(np.logical_or(samples > 1, samples < 0)).flatten()

        while len(oob_idx) > 0:
            samples[oob_idx] -= delta[oob_idx]  # revert move
            delta[oob_idx] = np.random.normal(size=len(oob_idx)) * self.bw
            samples[oob_idx] += delta[oob_idx]
            oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx] > 1, samples[oob_idx] < 0))].flatten()

        return samples


class AitchisonAitken(BaseKernel):
    def __call__(self, x_test):
        distances = np.rint(x_test[None, :] - self.data[:, None])

        idx = np.abs(distances) == 0
        distances[idx] = 1 - self.bw
        distances[~idx] = self.bw / (self.num_values - 1)

        return distances

    def sample(self, sample_indices=None, num_samples=1):
        """ returns samples according to the KDE

            Parameters
            ----------
                sample_indices: list of ints
                    Indices into the training data used as centers for the samples

                num_samples: int
                    if samples_indices is None, this specifies how many samples
                    are drawn.

        """
        if sample_indices is None:
            sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
        samples = self.data[sample_indices]

        samples = samples.squeeze()

        if self.num_values == 1:
            # handle cases where there is only one value!
            return samples

        probs = self.bw * np.ones(self.num_values) / (self.num_values - 1)
        probs[0] = 1 - self.bw

        delta = np.random.choice(self.num_values, size=num_samples, p=probs)
        samples = np.mod(samples + delta, self.num_values)

        return samples


class WangRyzinOrdinal(BaseKernel):

    def _compute_weights(self):
        if not self.fix_boundary:
            return tuple(1.)
        np.zeros(self.data.shape[0])
        self.weights = 1.
        x_test = np.arange(self.num_values)
        pdfs = self.__call__(x_test)
        weights = 1. / pdfs.sum(axis=1)[:, None]
        return weights

    def __call__(self, x_test):
        distances = x_test[None, :] - self.data[:, None]

        idx = np.abs(distances) < .1  # distances smaller than that are considered zero

        pdfs = np.zeros_like(distances, dtype=np.float)
        pdfs[idx] = (1 - self.bw)
        pdfs[~idx] = 0.5 * (1 - self.bw) * np.power(self.bw, np.abs(distances[~idx]))
        # reweigh to compensate for boundaries
        pdfs *= self.weights

        return pdfs

    def sample(self, sample_indices=None, num_samples=1):
        """ returns samples according to the KDE

            Parameters
            ----------
                sample_indices: list of ints
                    Indices into the training data used as centers for the samples

                num_samples: int
                    if samples_indices is None, this specifies how many samples
                    are drawn.

        """
        if sample_indices is None:
            sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
        samples = self.data[sample_indices]

        possible_steps = np.arange(-self.num_values + 1, self.num_values)
        idx = (np.abs(possible_steps) < 1e-2)

        ps = 0.5 * (1 - self.bw) * np.power(self.bw, np.abs(possible_steps))
        ps[idx] = (1 - self.bw)
        ps /= ps.sum()

        delta = np.zeros_like(samples)
        oob_idx = np.arange(samples.shape[0])

        while len(oob_idx) > 0:
            samples[oob_idx] -= delta[oob_idx]  # revert move
            delta[oob_idx] = np.random.choice(possible_steps, size=len(oob_idx), p=ps)
            samples[oob_idx] += delta[oob_idx]
            # import pdb; pdb.set_trace()
            oob_idx = oob_idx[
                np.argwhere(np.logical_or(samples[oob_idx] > self.num_values - 0.9, samples[oob_idx] < -0.1)).flatten()]
        return np.rint(samples)


class WangRyzinInteger(BaseKernel):
    def _compute_weights(self):
        if not self.fix_boundary:
            return tuple(1.)
        x_test = np.linspace(1 / (2 * self.num_values), 1 - (1 / (2 * self.num_values)), self.num_values, endpoint=True)
        self.weights = 1.
        pdfs = self.__call__(x_test)
        weights = 1. / pdfs.sum(axis=1)[:, None]
        return weights

    def __call__(self, x_test):
        distances = (x_test[None, :] - self.data[:, None])

        pdfs = np.zeros_like(distances, dtype=np.float)

        idx = np.abs(distances) < 1 / (3 * self.num_values)  # distances smaller than that are considered zero
        pdfs[idx] = (1 - self.bw)
        pdfs[~idx] = 0.5 * (1 - self.bw) * np.power(self.bw, np.abs(distances[~idx]) * self.num_values)
        # reweigh to compensate for boundaries
        pdfs *= self.weights

        return pdfs

    def sample(self, sample_indices=None, num_samples=1):
        """ returns samples according to the KDE

            Parameters
            ----------
                sample_indices: list of ints
                    Indices into the training data used as centers for the samples

                num_samples: int
                    if samples_indices is None, this specifies how many samples
                    are drawn.

        """
        if sample_indices is None:
            sample_indices = np.random.choice(self.data.shape[0], size=num_samples)
        samples = self.data[sample_indices]

        possible_steps = np.arange(-self.num_values + 1, self.num_values) / self.num_values
        ps = 0.5 * (1 - self.bw) * np.power(self.bw, np.abs(possible_steps))
        ps[self.num_values - 1] = (1 - self.bw)
        ps /= ps.sum()

        delta = np.zeros_like(samples)
        oob_idx = np.arange(samples.shape[0])

        while len(oob_idx) > 0:
            samples[oob_idx] -= delta[oob_idx]  # revert move
            delta[oob_idx] = np.random.choice(possible_steps, size=len(oob_idx), p=ps)
            samples[oob_idx] += delta[oob_idx]
            oob_idx = oob_idx[np.argwhere(np.logical_or(samples[oob_idx] > 1 - 1 / (3 * self.num_values),
                                                        samples[oob_idx] < 1 / (3 * self.num_values))).flatten()]

        return samples


# --------------------------------------------------- KDE -----------------------------------------------------------
class MultivariateKDE(AbstractEPM):
    """
    MultivariateKDE
    This implementation is based on the implementation of hpbandster:
    https://github.com/automl/HpBandSter/tree/master/hpbandster
    """

    def __init__(self, configspace, types: np.ndarray, bounds: typing.List[typing.Tuple[float, float]],
                 fully_dimensional=True, min_bandwidth=1e-4, fix_boundary=True):
        """
        Parameters:
        -----------
            configspace: ConfigSpace.ConfigurationSpace object
                description of the configuration space
            fully_dimensional: bool
                if True, a true multivariate KDE is build, otherwise it's approximated by
                the product of one dimensional KDEs

            min_bandwidth: float
                a lower limit to the bandwidths which can insure 'uncertainty'

        """
        super().__init__(types, bounds)
        self.configspace = configspace
        self.types, self.num_values = types, bounds
        self.min_bandwidth = min_bandwidth
        self.fully_dimensional = fully_dimensional
        self.fix_boundary = fix_boundary

        # precompute bandwidth bounds
        self.bw_bounds = []

        max_bw_cont = 0.5
        max_bw_cat = 0.999

        for t in self.types:
            if t == 'C':
                self.bw_bounds.append((min_bandwidth, max_bw_cont))
            else:
                self.bw_bounds.append((min_bandwidth, max_bw_cat))

        self.bw_clip = np.array([bwb[1] for bwb in self.bw_bounds])

        # initialize other vars
        self.bandwidths = np.array([float('NaN')] * len(self.types))
        self.kernels = []
        for t, n in zip(self.types, self.num_values):

            kwargs = {'num_values': n, 'fix_boundary': fix_boundary}

            if t == 'I':
                self.kernels.append(WangRyzinInteger(**kwargs))
            elif t == 'C':
                self.kernels.append(Gaussian(**kwargs))
            elif t == 'O':
                self.kernels.append(WangRyzinOrdinal(**kwargs))
            elif t == 'U':
                self.kernels.append(AitchisonAitken(**kwargs))
        self.data = None

    def _train(self, data: np.ndarray, weights: typing.Union[None, np.ndarray] = None,
               bw_estimator: str = 'scott', efficient_bw_estimation: bool = True,
               update_bandwidth: bool = True) -> 'AbstractEPM':
        """
            fits the KDE to the data by estimating the bandwidths and storing the data

            Parameters
            ----------
                data: 2d-array, shape N x M
                    N datapoints in an M dimensional space to which the KDE is fit
                weights: 1d array
                    N weights, one for every data point.
                    They will be normalized to sum up to one
                bw_estimator: str
                    allowed values are 'scott' and 'mlcv' for Scott's rule of thumb
                    and the maximum likelihood via cross-validation
                efficient_bw_estimation: bool
                    if true, start bandwidth optimization from the previous value, otherwise
                    start from Scott's values
                update_bandwidth: bool
                    whether to update the bandwidths at all
        """

        if self.data is None:
            # overwrite some values in case this is the first fit of the KDE
            efficient_bw_estimation = False
            update_bandwidth = True

        self.data = np.asfortranarray(data)
        for i, k in enumerate(self.kernels):
            self.kernels[i].data = self.data[:, i]

        self.weights = self._normalize_weights(weights)

        if not update_bandwidth:
            return self

        if not efficient_bw_estimation or bw_estimator == 'scott':
            # inspired by the the statsmodels code
            sigmas = np.std(self.data, ddof=1, axis=0)
            IQRs = np.subtract.reduce(np.percentile(self.data, [75, 25], axis=0))
            self.bandwidths = 1.059 * np.minimum(sigmas, IQRs) * np.power(self.data.shape[0], -0.2)
            # crop bandwidths for categorical parameters
            self.bandwidths = np.clip(self.bandwidths, self.min_bandwidth, self.bw_clip)

        if bw_estimator == 'mlcv':
            # optimize bandwidths here
            def opt_me(bw):
                self.bandwidths = bw
                self._set_kernel_bandwidths()
                return self._loo_negloglikelihood()

            res = spo.minimize(opt_me, self.bandwidths, bounds=self.bw_bounds, method='SLSQP')
            self.optimizer_result = res
            self.bandwidths[:] = res.x
        self._set_kernel_bandwidths()
        return self

    def _set_kernel_bandwidths(self) -> None:
        """
        TODO
        """
        for i, b in enumerate(self.bandwidths):
            self.kernels[i].set_bandwidth(b)

    def set_bandwidths(self, bandwidths) -> None:
        """
        Sets the bandwidth
        :param bandwidths:
        """
        self.bandwidths[:] = bandwidths
        self._set_kernel_bandwidths()

    def _normalize_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        Normalizes the weights
        :param weights: Weight array
        :return: Normalized weight array
        """
        weights = np.ones(self.data.shape[0]) if weights is None else weights
        weights /= weights.sum()

        return weights

    def _individual_pdfs(self, x_test: np.ndarray) -> np.ndarray:
        """
        TODO
        :param x_test:
        :return:
        """
        pdfs = np.zeros(shape=[x_test.shape[0], self.data.shape[0], self.data.shape[1]], dtype=np.float)

        for i, k in enumerate(self.kernels):
            pdfs[:, :, i] = k(x_test[:, i]).T

        return pdfs

    def _loo_negloglikelihood(self) -> np.float:
        """
        TODO
        :return:
        """
        # get all pdf values of the training data (including 'self interaction')
        pdfs = self._individual_pdfs(self.data)

        # get indices to remove diagonal values for LOO part :)
        indices = np.diag_indices(pdfs.shape[0])

        # combine values based on fully_dimensional!
        if self.fully_dimensional:
            pdfs[indices] = 0  # remove self interaction
            pdfs = np.prod(pdfs, axis=-1)

            # take weighted average (accounts for LOO!)
            lhs = np.sum(pdfs * self.weights, axis=-1) / (1 - self.weights)
        else:
            pdfs[indices] = 0  # we sum first so 0 is the appropriate value
            pdfs *= self.weights[:, None, None]

            pdfs = pdfs.sum(axis=-2) / (1 - self.weights[:, None])
            lhs = np.prod(pdfs, axis=-1)

        return -np.sum(self.weights * np.log(lhs))

    def _predict(self, X: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray[None]]:
        """
            Computes the probability density function at all x_test
        """
        N, D = self.data.shape
        x_test = np.asfortranarray(X)
        x_test = x_test.reshape([-1, D])

        pdfs = self._individual_pdfs(x_test)
        # import pdb; pdb.set_trace()
        # combine values based on fully_dimensional!
        if self.fully_dimensional:
            # first the product of the individual pdfs for each point in the data across dimensions and then the average
            # (factorized kernel)
            pdfs = np.sum(np.prod(pdfs, axis=-1) * self.weights[None, :], axis=-1)
        else:
            # first the average over the 1d pdfs and the the product over dimensions (TPE like factorization of the pdf)
            pdfs = np.prod(np.sum(pdfs * self.weights[None, :, None], axis=-2), axis=-1)
        return pdfs, np.full(pdfs.shape, None)

    def _sample(self, num_samples: int=1) -> np.ndarray:
        """
        TODO
        :param num_samples:
        :return:
        """

        samples = np.zeros([num_samples, len(self.types)], dtype=np.float)

        if self.fully_dimensional:
            sample_indices = np.random.choice(self.data.shape[0], size=num_samples)

        else:
            sample_indices = None

        for i, k in enumerate(self.kernels):
            samples[:, i] = k.sample(sample_indices, num_samples)

        return samples
