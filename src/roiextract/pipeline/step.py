class PipelineStep:
    """
    Base class for all pipeline steps that describe a linear transformation
    of the data.

    Attributes
    ----------
    prepared : bool
        Indicates whether the step (i.e., the underlying method) has been
        fit to the data.
    weights : array
        The weight matrix corresponding to the linear transformation defined
        by this pipeline step.
    row_names : list of str
        Names for rows of the weight matrix that corresponds to this step.
    """

    def __init__(self):
        self.prepared = False

    def check_if_prepared(self):
        """
        Checks if the pipeline step has been fit to data. To pass the check,
        the overriden version of the :meth:`fit()` method should set the
        ``prepared`` attribute to ``True``.

        Raises
        ------
        RuntimeError
            If the pipeline step has not been prepared (i.e., fit to data).
        """
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
        data
            The data to fit the step on. The expected type and shape of the data
            depends on the specific step implementation.
        **kwargs
            Additional keyword arguments that may be required for fitting the step.
            By default, no arguments are provided by the :class:`~roiextract.pipeline.ExtractionPipeline` class. The step
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
        data
            The data to transform. The expected type and shape of the data
            depends on the specific step implementation.

        Returns
        -------
        transformed_data
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
        data
            The data to fit and transform. The expected type and shape of the data
            depends on the specific step implementation.
        **kwargs
            Additional keyword arguments that may be required for fitting the step.
            By default, no additional arguments are provided. The step
            implementation can request specific arguments by overriding the
            :meth:`request_args()` method.

        Returns
        -------
        transformed_data
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
            The source space object.
        labels : list of Label
            The definitions labels (ROIs) to be used for extraction.
        subject : str, optional
            The subject identifier, as specified in FreeSurfer.
        subjects_dir : str, optional
            The directory containing FreeSurfer data.
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
    def weights(self):
        """
        The weight matrix corresponding to the linear transformation defined
        by this pipeline step.
        """
        raise NotImplementedError("weights property must be implemented by subclasses")

    @property
    def row_names(self):
        """
        Names for rows of the weight matrix that corresponds to this step.
        """
        return None
