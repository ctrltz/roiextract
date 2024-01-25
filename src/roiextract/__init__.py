try:
    from importlib.metadata import version

    __version__ = version("roiextract")
except Exception:
    __version__ = "Failed to determine the version"
