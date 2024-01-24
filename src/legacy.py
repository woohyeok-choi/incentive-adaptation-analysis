class LogisticRegressionIncentive(BaseIncentive):
    def __init__(
            self,
            incentives: np.ndarray,
            incentive_default: float,
            expected_likelihood: float,
            window_size: int = None,
            allow_exploration: bool = False,
            random_state: int = None,
            **kwargs
    ):
        super().__init__(incentives, random_state, **kwargs)

        self._incentive_default = incentive_default
        self._expected_likelihood = expected_likelihood
        self._window_size = window_size
        self._allow_exploration = allow_exploration

        self._lr = None
        self._encoder = None
        self._histories = []

    @property
    def coef_incentive_(self):
        if self._lr:
            return np.ravel(self._lr.coef_)[0]
        else:
            return np.nan

    def expect(self, incentive: float, context: any = None) -> float:
        assert self._lr is not None
        assert self._encoder is not None

        categories = self._encoder.categories_[0]
        if context not in categories:
            context = np.zeros((1, len(categories)))
        else:
            context = self._encoder.transform(np.asarray([[context]], dtype=object))

        incentive = np.asarray([[incentive]])
        X = np.concatenate((incentive, context), axis=1)
        return np.ravel(self._lr.predict_proba(X=X)[:, 1]).item(0)

    def choose(self, context: any = None) -> float:
        if self._lr is None or self._encoder is None:
            responses = np.asarray([i for _, _, i in self._histories])

            if len(responses) == 0:
                return self._incentive_default
            elif np.all(responses == 1.0):
                return np.min(self.incentives)
            elif np.all(responses == 0.0):
                return np.max(self.incentives)

        if self.coef_incentive_ < 1e-3:
            if self._allow_exploration:
                return self.random.choice(self.incentives)
            else:
                return np.min(self.incentives)
        probs = np.asarray([
            self.expect(incentive=incentive, context=context) for incentive in self.incentives
        ])
        diff = np.abs(probs - self._expected_likelihood)
        opt_incentives = self.incentives[np.where(diff == np.min(diff))]
        return self.random.choice(opt_incentives)

    def update(self, incentive: float, response: bool, context: any = None) -> Optional[Dict[str, any]]:
        self._histories.append((context, incentive, response))

        histories = self._histories[
                    -self._window_size:] if self._window_size is not None and self._window_size > 0 else self._histories

        responses = np.asarray([i for _, _, i in histories], dtype=int)
        contexts = np.asarray([[i] for i, _, _ in histories])
        incentives = np.asarray([[i] for _, i, _ in histories])

        if len(np.unique(responses)) == 1:
            return None

        self._encoder = OneHotEncoder(sparse_output=False, dtype=np.int32, drop=None).fit(contexts)

        X = np.concatenate((incentives, self._encoder.transform(contexts)), axis=1)
        self._lr = LogisticRegression(C=0.1, tol=1e-4, solver='lbfgs', penalty='l2').fit(X, responses)

        y_prob = self._lr.predict_proba(X)[:, 1]
        y_pred = self._lr.predict(X)

        return {'accuracy': accuracy_score(responses, y_pred), 'logloss': log_loss(responses, y_prob)}
