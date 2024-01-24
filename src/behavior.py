import numpy as np
from abc import abstractmethod, ABC
from typing import List
from numpy.typing import NDArray


class BaseBehavior(ABC):
    @abstractmethod
    def _likelihood(
        self, 
        compensation: float,
        rounds: int = None,
        context: int = None
    ) -> float:
        """
        Returns the likelihood (or probability) of behavior occurrences for a given compensation, round, and context.

        Parameters
        ----------
        compensation: float
            The compensation for the behavior change
        rounds: int, optional
            The number of rounds, interactions, or compensation suggestions
        context: int, optional
            The current context

        Returns
        ---------
        float
            The likelihood.
        """
        pass

    @abstractmethod
    def is_valid(self) -> bool:
        """
        Returns whether the current set of parameters are valid or not.

        Returns
        ---
        bool
            The validity of the current set of parameters.
        """
        pass

    def estimate_likelihood(
            self, compensation: float, rounds: int = None, context: int = None
    ) -> float:
        """
        This is the clipped version of the '_likelihood' function.
        """
        return np.clip(
            self._likelihood(compensation, rounds, context),
            a_min=0.0, a_max=1.0
        )

    def __str__(self):
        args = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        cls = self.__class__.__name__
        return f'{cls}({args})'

    def __repr__(self):
        args = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        cls = self.__class__.__name__
        return f'{cls}({args})'


class DecayedBehavior(BaseBehavior, ABC):
    def __init__(
        self, 
        decay_step: int = None,
        decay_factor: float = 1.0,
        decay_likelihood_min: float = 0
    ):
        """
        Parameters
        ----------
        decay_step: int, optional
            The number of unit rounds that a likelihood decays.
            For example, when decay_step = 10, the likelihood is multiplied by 'decay_factor' every ten rounds.
        decay_factor: float, optional
            The factor multiplied to the likelihood, ranging from 0 to 1.
            When decay_factor = 1, the likelihood does not decrease at all.
        decay_likelihood_min: float, optional
            The minimum likelihood
        """
        self.decay_step = decay_step
        self.decay_factor = decay_factor
        self.decay_likelihood_min = decay_likelihood_min
    
    def estimate_likelihood(
        self, compensation: float, rounds: int = None, context: int = None
    ) -> float:
        p = np.clip(
            self._likelihood(compensation, rounds, context),
            a_min=0.0, a_max=1.0
        )
        if self.decay_step and self.decay_factor:
            decay = np.power(self.decay_factor, rounds // self.decay_step)
            p = np.clip(p * decay, a_min=self.decay_likelihood_min, a_max=1.0)
        return p


class StaticBehavior(DecayedBehavior):
    def __init__(
        self,
        likelihood: float = 0,
        **kwargs
    ):
        """
        The behavior occurs at the 'likelihood' with no respect to the compensation.

        Parameters
        ----------
        likelihood: float
            The base likelihood of behavior occurrences

        """
        super().__init__(**kwargs)
        self.likelihood = likelihood

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        return self.likelihood

    def is_valid(self) -> bool:
        return 0 <= self.likelihood <= 1


class StepBehavior(DecayedBehavior):
    def __init__(
        self, 
        likelihood_0: float,
        likelihood_1: float,
        threshold: float,
        **kwargs
    ):
        """
        The behavior occurs at the 'likelihood_0' if a compensation is less than 'threshold';
        otherwise occurs at the 'likelihood_1'.

        Parameters
        ----------
        likelihood_0: float
            The likelihood of behavior occurrences when a compensation is less than 'threshold'.
        likelihood_1: float
            The likelihood of behavior occurrences when a compensation is equal to or greater than 'threshold'.
        threshold: float
            The threshold of the compensation.

        """
        super().__init__(**kwargs)
        self.likelihood_0 = likelihood_0
        self.likelihood_1 = likelihood_1
        self.threshold = threshold

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        return self.likelihood_0 if self.threshold >= compensation else self.likelihood_1

    def is_valid(self) -> bool:
        return 0 <= self.likelihood_0 <= 1 and 0 <= self.likelihood_1 <= 1


class RandomBehavior(DecayedBehavior):
    def __init__(
        self, 
        compensations: NDArray[float],
        random_state: int = None,
        **kwargs
    ) -> None:
        """
        The behavior randomly occurs at each compensation.

        Parameters
        ----------
        compensations: NDArray[float]
            The compensation
        random_state
            Random seed
        """
        super().__init__(**kwargs)
        self.compensations = np.asarray(compensations)
        self.random_state = random_state
        self._random = np.random.default_rng(random_state)
        self.likelihoods = self._random.uniform(low=0, high=1, size=len(self.compensations))

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        return self.likelihoods[np.argmin(np.abs(self.compensations - compensation))].item(0)

    def is_valid(self) -> bool:
        return True


class SigmoidBehavior(DecayedBehavior):
    def __init__(
            self,
            compensation_0: float,
            compensation_1: float,
            likelihood_0: float = 0.0,
            likelihood_1: float = 1.0,
            **kwargs
    ):
        """
        The behavior occurs by following the sigmoid function;
        there is no need that 'compensation_0' is greater than 'compensation_1' and 'likelihood_0' is greater than 'likelihood_1'.

        Because the sigmoid function does not give the exact zero or one value,
        it will give the approximate value of the likelihood:
        - the likelihood_0 * 0.001 at a compensation_0
        - the likelihood_0 * 0.001 + likelihood_1 * 0.999 at a compensation_1

        Parameters
        ----------
        compensation_0: float
            The compensation that the behavior occurs with the likelihood of 'likelihood_0'.
        compensation_1: float
            The compensation that the behavior occurs with the likelihood of 'likelihood_1'.
        likelihood_0: float
            The likelihood that the behavior occurs when the `compensation_0` is given.
        likelihood_1: float
            The likelihood that the behavior occurs when the `compensation_1` is given.
        """
        super().__init__(**kwargs)
        self.compensation_0 = compensation_0
        self.compensation_1 = compensation_1
        self.likelihood_0 = likelihood_0
        self.likelihood_1 = likelihood_1

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        mi, ma = min(self.compensation_0, self.compensation_1), max(self.compensation_0, self.compensation_1)
        compensation = np.clip(compensation, a_min=mi, a_max=ma)
        l = np.log(999) * (2 * compensation - self.compensation_0 - self.compensation_1) / (self.compensation_1 - self.compensation_0)
        l = 1 / (1 + np.exp(-l))
        return self.likelihood_0 + (self.likelihood_1 - self.likelihood_0) * l

    def is_valid(self) -> bool:
        return True


class LinearBehavior(DecayedBehavior):
    def __init__(
            self,
            compensation_0: float,
            compensation_1: float,
            likelihood_0: float = 0.0,
            likelihood_1: float = 1.0,
            **kwargs
    ):
        """
        The behavior occurs by following the linear function;
        there is no need that 'compensation_0' is greater than 'compensation_1' and 'likelihood_0' is greater than 'likelihood_1'.

        Parameters
        ----------
        compensation_0: float
            The compensation that the behavior occurs with the likelihood of 'likelihood_0'.
        compensation_1: float
            The compensation that the behavior occurs with the likelihood of 'likelihood_1'.
        likelihood_0: float
            The likelihood that the behavior occurs when the `compensation_0` is given.
        likelihood_1: float
            The likelihood that the behavior occurs when the `compensation_1` is given.
        """
        super().__init__(**kwargs)
        self.compensation_0 = compensation_0
        self.compensation_1 = compensation_1
        self.likelihood_0 = likelihood_0
        self.likelihood_1 = likelihood_1

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        mi, ma = min(self.compensation_0, self.compensation_1), max(self.compensation_0, self.compensation_1)
        compensation = np.clip(compensation, a_min=mi, a_max=ma)
        return self.likelihood_0 + (compensation - self.compensation_0) * (self.likelihood_1 - self.likelihood_0) / (self.compensation_1 - self.compensation_0)

    def is_valid(self) -> bool:
        return True


class ContextDependentBehavior(BaseBehavior):
    def __init__(
        self,
        n_contexts: int,
        behaviors: List[BaseBehavior],
        **kwargs
    ) -> None:
        """
        The different behavior across contexts.

        Parameters
        ----------
        n_contexts: int
            The number of contexts. This should be the same as the length of 'behaviors'.
        behaviors: List[BaseBehavior]
            The list of behaviors whose length is equal to the `n_contexts`.

        """
        super().__init__()
        self.n_contexts = n_contexts
        self.behaviors = behaviors

    def is_valid(self) -> bool:
        return len(self.behaviors) == self.n_contexts

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        behavior = self.behaviors[context]
        return behavior.estimate_likelihood(compensation, rounds, context)


class DynamicBehavior(BaseBehavior):
    def __init__(
        self,
        rounds: NDArray[int],
        behaviors: List[BaseBehavior],
        **kwargs
    ) -> None:
        """
        The different behavior across round indices.
        Suppose rounds is [5, 10, 15] and the behaviors = [A, B, C, D].
        - Rounds 4 or before: the behavior A
        - Rounds 5 to 9: the behavior B
        - Rounds 10 to 14: the behavior C
        - Rounds 15 or later: the behavior D

        Parameters
        ----------
        rounds: NDArray[int]
            An array of round indices indicating the which behavior is to be considered.
            Its length should be one longer than the length of 'behaviors'.
        behaviors: List[BaseBehavior]
            The list of different behaviors.
        """
        super().__init__()
        self.rounds = np.asarray(rounds)
        self.behaviors = behaviors

    def is_valid(self) -> bool:
        return len(self.rounds) == len(self.behaviors) - 1

    def _likelihood(self, compensation: float, rounds: int = None, context: int = None) -> float:
        i_behavior = np.digitize(rounds, bins=self.rounds)
        behavior = self.behaviors[i_behavior]
        return behavior.estimate_likelihood(compensation, rounds, context)
