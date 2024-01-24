import numpy as np
from numpy.typing import NDArray
from abc import abstractmethod, ABC
from typing import Dict, Optional, Callable, Union, Tuple
import copy

class BaseIncentive(ABC):
    def __init__(
            self,
            compensations: NDArray[float],
            random_state: int = None,
            **kwargs
    ) -> None:
        """
        Abstract base class for the incentive strategy

        Parameters
        ----------
        compensations: NDArray[float]
            An array of compensations that will be given to the user for their behavior change.
        random_state:
            Random seed
        """
        self.random_state = random_state
        self._compensations = np.asarray(compensations)
        self._random = np.random.default_rng(random_state)

    @abstractmethod
    def choose(self, context: int = None) -> Tuple[float, Optional[Dict]]:
        """
        It returns the best compensation and optionally gives its corresponding information.

        Parameters
        ----------
        context: int
            A context that needs to be considered to estimate the best compensation

        Returns
        ---
        compensation, information: Tuple[float, Optional[Dict]]
            'compensation' is the best compensation;
            'information' is the optional information explaining why such a compensation is estimated.
        """
        pass

    @abstractmethod
    def update(self, compensation: float, response: bool, context: int = None):
        """
        It updates the compensation estimation.

        Parameters
        ----------
        compensation: float
            The compensation that was suggested to the user.
        response: bool
            Whether the user accepts the compensation and changes his/her behavior.
        context:
            A context that such a compensation and a response are observed.
        """
        pass

    def __str__(self):
        args = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        cls = self.__class__.__name__
        return f'{cls}({args})'

    def __repr__(self):
        args = ', '.join([f'{k}={v}' for k, v in self.__dict__.items() if not k.startswith('_')])
        cls = self.__class__.__name__
        return f'{cls}({args})'


class StaticIncentive(BaseIncentive):
    def __init__(
            self,
            base_compensation: float,
            **kwargs
    ) -> None:
        """
        It gives the same amount of compensations everytime.

        Parameters
        ----------
        base_compensation: float
            The amount of compensations that will be suggested.
        """
        super().__init__(**kwargs)
        self.base_compensation = base_compensation

    def choose(self, context: int = None) -> Tuple[float, Optional[Dict]]:
        return self.base_compensation, None

    def update(self, compensation: float, response: bool, context: int = None):
        return


class RandomIncentive(BaseIncentive):
    def __init__(
            self,
            **kwargs
    ) -> None:
        """
        It uniformly and randomly chooses the compensation among all possible compensations.
        """
        super().__init__(**kwargs)

    def choose(self, context: int = None) -> Tuple[float, Optional[Dict]]:
        return self._random.choice(self._compensations), {i: (1.0 / len(self._compensations)) for i in self._compensations}

    def update(self, compensation: float, response: bool, context: int = None):
        return


class ThompsonSamplingIncentive(BaseIncentive):
    def __init__(
            self,
            w: float = 1.0,
            decay_factor: float = 1.0,
            **kwargs
    ) -> None:
        """
        It chooses the optimal compensation that maximizes the expected acceptance or adherence rate using Thompson sampling.

        Parameters
        ----------
        w: float, default = 1.0
            The weight that is summed into the alpha or beta parameter of the Beta distribution every update.
            The larger 'w' leads to growing trials, meaning faster exploitation.
        decay_factor: float, default = 1.0
            The weight multiplied to the alpha and beta parameters of the Beta distribution every update.
            This parameter is used to decay the effect of the past trials.
            The decay_factor equals to 1.0 means that there is no decay.
        """
        super().__init__(**kwargs)
        self.w = w
        self.decay_factor = decay_factor
        self._alpha = np.ones(len(self._compensations)) * self.w
        self._beta = np.ones(len(self._compensations)) * self.w

    def choose(self, context: int = None) -> Tuple[float, Optional[Dict]]:
        E_success, info = list(), dict()
        for a, b, c in zip(self._alpha, self._beta, self._compensations):
            e = self._random.beta(a, b)
            E_success.append(e)
            info[f'{c}_alpha'] = a
            info[f'{c}_beta'] = b
            info[f'{c}_success'] = e
        return self._compensations[np.argmax(E_success)].item(0), info

    def update(self, compensation: float, response: bool, context: int = None):
        self._alpha = np.clip(self._alpha * self.decay_factor, a_min=self.w, a_max=None)
        self._beta = np.clip(self._beta * self.decay_factor, a_min=self.w, a_max=None)

        idx = np.flatnonzero(self._compensations == compensation).item(0)
        self._alpha[idx] = self._alpha[idx] + (self.w if response else 0)
        self._beta[idx] = self._beta[idx] + (0 if response else self.w)


class MOThompsonSamplingIncentive(ThompsonSamplingIncentive):
    def __init__(
            self,
            frame: str = 'gain',
            **kwargs
    ) -> None:
        """
        It chooses the optimal compensation that
        - Maximizes the expected acceptance rate
        - Minimizes the expected compensation will be given to the user

        Parameters
        ----------
        frame: str, default = 'gain'
            This parameter indicates how the compensation is framed to the user.
            If it sets to 'gain', the compensation is given to the users when they change their behaviors.
            Otherwise, the penalty is deducted from the individual budget when they decline to change their behaviors.
        """
        super().__init__(**kwargs)
        if frame is None or frame not in ('gain', 'loss'):
            raise ValueError('the argument, "frame", should be one of "gain" or "loss".')
        self.frame = frame

    def choose(self, context: int = None) -> Tuple[float, Optional[Dict]]:
        E_success, E_cost, info = list(), list(), dict()
        for a, b, c in zip(self._alpha, self._beta, self._compensations):
            e_success = self._random.beta(a, b)
            e_cost = e_success * c if self.frame == 'gain' else (1 - e_success) * c
            E_success.append(e_success)
            E_cost.append(e_cost)
            info[f'{c}_alpha'] = a
            info[f'{c}_beta'] = b
            info[f'{c}_success'] = e_success
            info[f'{c}_cost'] = e_cost

        if self.frame == 'gain':
            direction = ('max', 'min')
        else:
            direction = ('max', 'max')
        E = np.column_stack((E_success, E_cost))
        I_opt = self._random.choice(self._find_pareto_frontiers(E, direction))
        compensation = self._compensations[I_opt].item(0)
        return compensation, info

    @classmethod
    def _is_dominated(cls, u: int, v: int, estimates: NDArray, direction: tuple):
        """
        At least one objective is better than another, 
        and other objectives are equal to or better than others.
        """
        n_objectives = estimates.shape[1]

        for i in np.arange(n_objectives):
            if direction[i] == 'max':
                is_dominated = estimates[u, i] > estimates[v, i]
            else:
                is_dominated = estimates[u, i] < estimates[v, i]

            for j in np.arange(n_objectives):
                if i == j:
                    continue
                else:
                    if direction[j] == 'max':
                        is_dominated = is_dominated and estimates[u, j] >= estimates[v, j]
                    else:
                        is_dominated = is_dominated and estimates[u, j] <= estimates[v, j]

            if is_dominated:
                return True

        return False

    @classmethod
    def _is_incomparable(cls, u: int, v: int, estimates: NDArray, direction: tuple):
        """
        At least one objective is better than another, 
        but at least another one objective is worse than others.
        """
        n_objectives = estimates.shape[1]

        for i in np.arange(n_objectives):
            if direction[i] == 'max':
                is_dominate = estimates[u, i] > estimates[v, i]
            else:
                is_dominate = estimates[u, i] < estimates[v, i]

            for j in np.arange(n_objectives):
                if direction[j] == 'max':
                    if i != j and estimates[u, j] < estimates[v, j] and is_dominate:
                        return True
                else:
                    if i != j and estimates[u, j] > estimates[v, j] and is_dominate:
                        return True
        return False

    @classmethod
    def _find_pareto_frontiers(cls, estimates: NDArray, direction: tuple):
        """
        One objective is dominated or incomparable toward all other objectives,
        such objective is the pareto frontier.
        """
        frontiers = []
        n = estimates.shape[0]

        for i in np.arange(n):
            is_pareto_frontier = True

            for j in np.arange(n):
                if i == j:
                    continue
                else:
                    is_dominated = cls._is_dominated(i, j, estimates, direction)
                    is_incomparable = cls._is_incomparable(i, j, estimates, direction)
                    is_pareto_frontier = is_pareto_frontier and (is_dominated or is_incomparable)

            if is_pareto_frontier:
                frontiers.append(i)

        return np.asarray(frontiers)


class ContextualIncentive(BaseIncentive):
    def __init__(
            self,
            incentives: Union[Dict[int, BaseIncentive], Callable[[Optional[int]], BaseIncentive], BaseIncentive],
            **kwargs
    ):
        """
        It differently estimates the compensation across contexts.

        Parameters
        ----------
        incentives
            The dictionary of incentive strategies where the key is a context and the value is the incentive strategy object (i.e., BaseIncentive);
            or, the function that takes the context and return the incentive strategy object.
        """
        super().__init__(**kwargs)
        self.incentives = incentives
        if type(incentives) is Dict:
            self._context_incentives = dict(incentives)
        else:
            self._context_incentives = dict()

    def choose(self, context: int = None) -> Tuple[float, Optional[Dict]]:
        if context not in self._context_incentives:
            if isinstance(self.incentives, Callable):
                self._context_incentives[context] = self.incentives(context)
            elif isinstance(self.incentives, BaseIncentive):
                self._context_incentives[context] = copy.copy(self.incentives)
        return self._context_incentives[context].choose(context)

    def update(self, compensation: float, response: bool, context: int = None):
        self._context_incentives[context].update(compensation, response, context)
