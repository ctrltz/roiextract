import mne
import numpy as np
import typing as T

from roiextract.filter import SpatialFilter
from roiextract.pipeline.step import PipelineStep


class ExtractionPipeline:
    """
    This class represents a pipeline for extracting ROI time courses. The pipeline
    should consist of one or more linear transformation steps.

    Parameters
    ----------
    steps : list of PipelineStep
        The list of pipeline steps (instances of :class:`PipelineStep` or its subclasses)
        that define the transformations to be applied to the data. The steps are applied
        in the order they appear in the provided list.

    Attributes
    ----------
    prepared : bool
        Indicates whether the pipeline has been fit to the data.
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        if not isinstance(steps, list) or not steps:
            raise ValueError("Expected at least one step in the pipeline.")

        if not all(isinstance(step, PipelineStep) for step in steps):
            raise ValueError(
                "All steps should be instances of a subclass of PipelineStep."
            )

        self.steps = steps
        self.prepared = False

    def __len__(self) -> int:
        return len(self.steps)

    def __repr__(self) -> str:
        step_reprs = [repr(step) for step in self.steps]
        return f"ExtractionPipeline <{len(self)} steps: {', '.join(step_reprs)}>"

    def _check_if_prepared(self) -> None:
        if not self.prepared:
            raise RuntimeError("The pipeline has not been prepared. Call fit() first.")

    def fit(
        self,
        data: mne.io.BaseRaw,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
        **kwargs: T.Any,
    ) -> "ExtractionPipeline":
        """
        Fit the pipeline to the provided data. This method should be called before
        calling :meth:`transform()`.

        Parameters
        ----------
        data : Raw
            The raw data to fit the pipeline on.
        src : SourceSpaces
            The definition of the considered source space for inverse modeling.
        labels : list of Label
            The list of ROIs for which time courses should be extracted.
        subject : str, optional
            The subject for which the pipeline is being fit. Currently, this argument
            is only used by centroid-based aggregation to compute the center of mass
            of the ROIs.
        subjects_dir : str, optional
            The directory containing the subjects' MRI data. Currently, this argument
            is only used by centroid-based aggregation to compute the center of mass
            of the ROIs.
        **kwargs
            Additional keyword arguments that may be required for fitting the pipeline.

        Returns
        -------
        self : ExtractionPipeline
            The fitted pipeline.

        Notes
        -----
        Each step in the pipeline receives only a subset of the provided arguments,
        depending on the specific requirements of the step. Custom steps can request
        specific arguments by overriding the :meth:`PipelineStep._request_args()`
        method.
        """
        for idx, step in enumerate(self.steps):
            step_args = step._request_args(src, labels, subject, subjects_dir, **kwargs)

            if idx < len(self) - 1:
                data = step.fit_transform(data, **step_args)
            else:
                step.fit(data, **step_args)
        self.prepared = True

        return self

    def transform(self, data: mne.io.BaseRaw) -> T.Any:
        """
        Transform the provided data using the fitted pipeline. For built-in steps,
        the transformation either executes the corresponding function in MNE-Python
        directly or matches it as closely as possible.

        Parameters
        ----------
        data : Raw
            The raw data to transform.

        Returns
        -------
        transformed_data
            The transformed data obtained by applying the pipeline to the input data.
        """
        self._check_if_prepared()
        for step in self.steps:
            data = step.transform(data)
        return data

    def fit_transform(
        self,
        data: mne.io.BaseRaw,
        src: mne.SourceSpaces,
        labels: mne.Label | list[mne.Label],
        subject: str | None = None,
        subjects_dir: str | None = None,
        **kwargs: T.Any,
    ) -> T.Any:
        """
        Fit the pipeline to the provided data and then apply the transformation.
        For the parameters and return values, see :meth:`fit()` and :meth:`transform()`,
        respectively.
        """
        self.fit(
            data, src, labels, subject=subject, subjects_dir=subjects_dir, **kwargs
        )
        return self.transform(data)

    def get_weights(self) -> np.ndarray:
        """
        The weight matrix corresponding to the linear transformation defined by the
        entire pipeline. It is obtained by multiplying the weight matrices of the
        individual steps in the pipeline.

        Returns
        -------
        weights : array
            The weight matrix corresponding to the linear transformation defined by the
            entire pipeline.
        """
        self._check_if_prepared()
        weights = self.steps[0].get_weights()
        for step in self.steps[1:]:
            weights = step.get_weights() @ weights
        return weights

    def get_names(self) -> list[str]:
        """
        Returns the names for rows of the resulting weight matrix.
        The names are taken from the last step in the pipeline.

        Returns
        -------
        row_names : list of str
            Names for rows of the weight matrix that corresponds to the entire pipeline.
        """
        self._check_if_prepared()
        return self.steps[-1].get_names()

    def get_filters(self) -> list[SpatialFilter]:
        """
        Get the list of spatial filters that represent the final data transformation
        for each considered ROI. The filters are generated using the values of
        :meth:`get_weights` and :meth:`get_names` methods of the pipeline, as well as
        :meth:`~PipelineStep.get_params` method of each step.

        Returns
        -------
        filters : list of SpatialFilter
            The resulting list of spatial filters.
        """
        self._check_if_prepared()
        method_params = {repr(step): step.get_params() for step in self.steps}
        row_names = self.get_names()
        weights = self.get_weights()
        if row_names is None:
            row_names = [""] * weights.shape[0]

        filters = []
        for w, name in zip(weights, row_names):
            filters.append(
                SpatialFilter(
                    w,
                    method="ExtractionPipeline",
                    method_params=method_params,
                    name=name,
                )
            )

        return filters
