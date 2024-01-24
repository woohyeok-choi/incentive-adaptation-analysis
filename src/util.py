from behavior import BaseBehavior
from incentive import BaseIncentive
import altair as alt
import numpy as np
import pandas as pd


def vis_behavior(
        behavior: BaseBehavior, incentive_min: float, incentive_max: float,
        rounds: int = None, context: int = None
):
    x = np.linspace(incentive_min, incentive_max, 1000)
    y = [behavior.estimate_likelihood(xx, rounds, context) for xx in x]
    df = pd.DataFrame(dict(x = x, y = y))
    return alt.Chart(df).mark_line().encode(
        x=alt.X('x:Q').title('Incentive'),
        y=alt.Y('y:Q').title('Likelihood')
    )


import pickle
import ray
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import scipy.stats as st
from functools import wraps
from contextlib import contextmanager


def load(path: str):
    with open(path, mode='rb') as f:
        return pickle.load(f)


def dump(obj, path: str):
    with open(path, mode='wb') as f:
        pickle.dump(obj, f)


def log(msg: any):
    print('[{}] {}'.format(datetime.now().strftime('%y-%m-%d %H:%M:%S'), msg))


@contextmanager
def on_ray(*args, **kwargs):
    try:
        if ray.is_initialized():
            ray.shutdown()
        ray.init(*args, **kwargs)
        yield None
    finally:
        ray.shutdown()
