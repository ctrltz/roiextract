class ExtractionPipeline:
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    def __repr__(self):
        step_reprs = [repr(step) for step in self.steps]
        return f"ExtractionPipeline <{len(self)} steps: {', '.join(step_reprs)}>"

    def fit(self, data, src, labels, subject=None, subjects_dir=None, **kwargs):
        for step in self.steps:
            step_args = step.request_args(src, labels, subject, subjects_dir, **kwargs)
            data = step.fit_transform(data, **step_args)
        return self

    def transform(self, data):
        for step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(
        self, data, src, labels, subject=None, subjects_dir=None, **kwargs
    ):
        self.fit(
            data, src, labels, subject=subject, subjects_dir=subjects_dir, **kwargs
        )
        return self.transform(data)

    @property
    def weights(self):
        weights = None
        for step in self.steps:
            if weights is None:
                weights = step.weights
            else:
                weights = step.weights @ weights
        return weights

    @property
    def row_names(self):
        """
        Returns the names for rows of the resulting weight matrix.
        The names are taken from the last step in the pipeline that
        has them defined (i.e., not set to None as per default).
        """
        step_idx = len(self) - 1
        while step_idx >= 0:
            row_names = self.steps[step_idx].row_names
            if row_names is not None:
                return row_names
            step_idx -= 1

        return None
