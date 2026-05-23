import numpy as np
import pytest

from roiextract.pipeline import ExtractionPipeline, PipelineStep


def test_extraction_pipeline_no_steps():
    with pytest.raises(ValueError):
        ExtractionPipeline(steps=[])


def test_extraction_pipeline_invalid_step():
    with pytest.raises(ValueError):
        ExtractionPipeline(steps=[1, 2, 3])


class StepRequestingArgs(PipelineStep):
    def __init__(self):
        super().__init__()

    def fit(self, data, src, labels, kwarg=None):
        assert src == "src"
        assert labels == "labels"
        assert kwarg == "kwarg"
        return self

    def transform(self, data):
        return None

    def _request_args(self, src, labels, subject=None, subjects_dir=None, **kwargs):
        return dict(src=src, labels=labels, kwarg=kwargs.get("kwarg", None))


def test_extraction_pipeline_provides_requested_args():
    pipeline = ExtractionPipeline(steps=[StepRequestingArgs()])
    pipeline.fit(data=None, src="src", labels="labels", kwarg="kwarg")


class Step1(PipelineStep):
    def __init__(self):
        super().__init__()

    def __repr__(self):
        return "Step1"

    def fit(self, data):
        self.prepared = True
        return self

    def transform(self, data):
        return data + 1

    @property
    def weights(self):
        return np.array([[1, 1], [1, 0]])


class Step2(PipelineStep):
    def __init__(self, raise_error_on_transform=False):
        super().__init__()
        self.raise_error_on_transform = raise_error_on_transform

    def __repr__(self):
        return "Step2"

    def fit(self, data, raise_error=False):
        self.prepared = True
        return self

    def transform(self, data):
        if self.raise_error_on_transform:
            raise ValueError("Raising error in case fit should not have been called.")

        return data * 2

    def _request_args(self, src, labels, subject=None, subjects_dir=None, **kwargs):
        return dict(raise_error=kwargs.get("raise_error", False))

    @property
    def weights(self):
        return np.array([[1, -1]])

    @property
    def row_names(self):
        return ["Row1"]

    @property
    def params(self):
        return dict(raise_error=self.raise_error_on_transform)


def test_extraction_pipeline_repr():
    pipeline = ExtractionPipeline(steps=[Step1(), Step2()])
    assert repr(pipeline) == "ExtractionPipeline <2 steps: Step1, Step2>"


def test_extraction_pipeline_fit():
    step1 = Step1()
    step2 = Step2(raise_error_on_transform=True)
    pipeline = ExtractionPipeline(steps=[step1, step2])

    # NOTE: Step2.transform() raises an error but it should not be called
    pipeline.fit(data=1, src=None, labels=None)

    assert step1.prepared
    assert step2.prepared
    assert pipeline.prepared


def test_extraction_pipeline_fit_transform():
    pipeline = ExtractionPipeline(steps=[Step1(), Step2()])
    pipeline.fit(data=1, src=None, labels=None)

    assert pipeline.transform(1) == 4
    assert pipeline.fit_transform(1, src=None, labels=None) == 4


def test_extraction_pipeline_weights():
    pipeline = ExtractionPipeline(steps=[Step1(), Step2()])
    pipeline.fit(data=1, src=None, labels=None)

    assert np.array_equal(pipeline.weights, np.array([[0, 1]]))


def test_extraction_pipeline_row_names():
    pipeline = ExtractionPipeline(steps=[Step1(), Step2()])
    pipeline.fit(data=1, src=None, labels=None)

    assert pipeline.row_names == ["Row1"]


def test_extraction_pipeline_get_filters():
    pipeline = ExtractionPipeline(steps=[Step1(), Step2()])
    pipeline.fit(data=1, src=None, labels=None)

    filters = pipeline.get_filters()
    assert len(filters) == 1
    assert filters[0].method == "ExtractionPipeline"
    assert np.array_equal(filters[0].w, np.array([0, 1]))
    assert filters[0].name == "Row1"
    assert filters[0].method_params == {"Step1": {}, "Step2": {"raise_error": False}}


def test_extraction_pipeline_check_if_prepared():
    pipeline = ExtractionPipeline(steps=[Step1(), Step2()])

    with pytest.raises(RuntimeError):
        pipeline.transform(1)

    with pytest.raises(RuntimeError):
        pipeline.weights

    with pytest.raises(RuntimeError):
        pipeline.row_names

    with pytest.raises(RuntimeError):
        pipeline.get_filters()
