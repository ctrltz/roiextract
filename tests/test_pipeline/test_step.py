import pytest

from roiextract.pipeline.step import PipelineStep


class MyStep(PipelineStep):
    def __init__(self):
        super().__init__()


def test_pipeline_step_check_if_prepared():
    step = MyStep()

    with pytest.raises(RuntimeError):
        step.check_if_prepared()

    step.prepared = True
    step.check_if_prepared()  # Should not raise an error
