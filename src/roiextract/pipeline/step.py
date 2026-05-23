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

    def fit(self, data, **kwargs):
        """
        Fit the underlying method to the provided data.
        This method should be called before calling :meth:`transform()`.

        Parameters
        ----------
        data : array-like
            The data to fit the step on. The expected type and shape of the data
            depends on the specific step implementation.
        **kwargs
            Additional keyword arguments that may be required for fitting the step.
            By default, no additional arguments are provided. The step
            implementation can request specific arguments by overriding the
            :meth:`request_args()` method.
        """
        raise NotImplementedError("fit() method must be implemented by subclasses")

    def transform(self, data):
        """
        Apply the transformation defined by this pipeline step to the
        provided data.

        Parameters
        ----------
        data : array-like
            The data to transform. The expected type and shape of the data
            depends on the specific step implementation.

        Returns
        -------
        transformed_data : array-like
            The transformed data. The type and shape of the returned data depend on
            the specific step implementation.
        """
        raise NotImplementedError(
            "transform() method must be implemented by subclasses"
        )

    def fit_transform(self, data, **kwargs):
        """
        Fit the step to the provided data and then apply the transformation.

        Parameters
        ----------
        data : array-like
            The data to fit and transform. The expected type and shape of the data
            depends on the specific step implementation.
        **kwargs
            Additional keyword arguments that may be required for fitting the step.
            By default, no additional arguments are provided. The step
            implementation can request specific arguments by overriding the
            :meth:`request_args()` method.

        Returns
        -------
        transformed_data : array-like
            The transformed data. The type and shape of the returned data depend on
            the specific step implementation.
        """
        return self.fit(data, **kwargs).transform(data)

    def request_args(self, src, labels, subject=None, subjects_dir=None, **kwargs):
        """
        Request additional arguments for the step.

        Parameters
        ----------
        src : SourceSpaces
            The source data.
        labels : list
            The labels for the data.
        subject : str, optional
            The subject identifier.
        subjects_dir : str, optional
            The directory containing subject data.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        args : dict
            A dictionary of keyword arguments to be passed to the :meth:`fit()`
            method.
        """
        return {}

    @property
    def row_names(self):
        return None
