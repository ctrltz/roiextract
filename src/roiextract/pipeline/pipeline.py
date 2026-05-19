from enum import Enum, auto


class StepType(Enum):
    Unknown = auto()
    SourceReconstruction = auto()
    ROIAggregation = auto()
    Orthogonalization = auto()


class PipelineStep:
    kind = StepType.Unknown

    def __init__(self):
        self.prepared = False

    def _check_if_prepared(self):
        if not self.prepared:
            raise RuntimeError(
                "The pipeline step has not been prepared. Call fit() first."
            )

    @property
    def row_names(self):
        return None


class ExtractionPipeline:
    def __init__(self, steps):
        self.steps = steps

    def __len__(self):
        return len(self.steps)

    @property
    def row_names(self) -> list[str] | None:
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
