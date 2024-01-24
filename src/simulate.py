from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Callable
from collections import defaultdict
from typing import ClassVar


@dataclass
class Simulation:
    incentive: BaseIncentive
    behavior: BaseBehavior
    contexts: NDArray[int]
    compensations: NDArray[float]
    responses: NDArray[bool]
    information: Dict[any, NDArray[any]]
    rounds: int
    successes: int
    costs: float
    name: str = None
    random_state: int = None
    prob_contexts: NDArray[float] = None
    _actor: ClassVar = ray.remote(Simulation.simulate)

    @staticmethod
    def simulate(
            incentive: BaseIncentive,
            behavior: BaseBehavior,
            max_rounds: int,
            name: str = None,
            random_state = None,
            prob_contexts: NDArray[float] = None,
            early_stop_success: int = None,
            early_stop_cost: float = None
    ):
        random = np.random.default_rng(random_state)
        contexts, compensations, responses, information = [], [], [], defaultdict(list)
        successes, costs, rounds = 0, 0, 0

        if prob_contexts is not None:
            prob_contexts = np.asarray(prob_contexts) / np.sum(prob_contexts)
            idx_contexts = np.arange(len(prob_contexts))
        else:
            idx_contexts = None

        for _ in range(max_rounds):
            rounds += 1
            context = random.choice(idx_contexts, p=prob_contexts) if prob_contexts is not None else None
            compensation, info = incentive.choose(context=context)
            threshold = behavior.estimate_likelihood(compensation=compensation, rounds=rounds, context=context)
            response = random.uniform() < threshold
            info = info or dict()

            contexts.append(context)
            compensations.append(compensation)
            responses.append(response)
            for k, v in info.items():
                information[k].append(v)

            successes += (1 if response else 0)
            costs += (compensation if response else 0)

            if early_stop_success and early_stop_success <= successes:
                break

            if early_stop_cost and early_stop_cost <= costs:
                break

            incentive.update(compensation=compensation, response=response, context=context)

        return Simulation(
            incentive=incentive,
            behavior=behavior,
            contexts=np.asarray(contexts),
            compensations=np.asarray(compensations),
            responses=np.asarray(responses),
            information={k: np.asarray(v) for k, v in information.items()},
            rounds=rounds,
            successes=successes,
            costs=costs,
            random_state=random_state,
            prob_contexts=prob_contexts,
            name=name
        )

    @staticmethod
    async def simulate_async(**kwargs):
        return Simulation.simulate(**kwargs)

    @staticmethod
    async def simulate_async_remote(**kwargs):
        return await Simulation._actor.remote(**kwargs)