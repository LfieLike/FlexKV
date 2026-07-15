# External backend adapters for FlexKV. Keep imports lazy so lightweight
# config/layout consumers do not require the compiled CUDA extension.
__all__ = ["MooncakeStoreConfig", "MooncakeStoreClient", "MooncakeStoreCacheEngine"]


def __getattr__(name):
    if name in __all__:
        from . import mooncake_store_utils as _m
        return getattr(_m, name)
    raise AttributeError(name)
