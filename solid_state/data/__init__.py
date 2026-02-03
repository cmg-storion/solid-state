from .datasampler import DataSampler

__all__ = [
    "DataSampler",
]

def __getattr__(name):
    if name == "DataSampler":
        from .datasampler import DataSampler
        return DataSampler
    raise AttributeError(f"module {__name__} has no attribute {name}")
