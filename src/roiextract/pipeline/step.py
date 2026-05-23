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
