from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType
from typing import Generic, TypeVar


def install_weight_transfer_fake() -> None:
    import sys

    TInitInfo = TypeVar("TInitInfo")
    TUpdateInfo = TypeVar("TUpdateInfo")

    @dataclass
    class WeightTransferInitInfo(ABC):
        pass

    @dataclass
    class WeightTransferUpdateInfo(ABC):
        pass

    class WeightTransferEngine(ABC, Generic[TInitInfo, TUpdateInfo]):
        init_info_cls: type
        update_info_cls: type
        supports_draft_weight_update: bool = True

        def __init__(self, config, vllm_config, device, model) -> None:
            self.config = config
            self.vllm_config = vllm_config
            self.parallel_config = vllm_config.parallel_config
            self.model_config = vllm_config.model_config
            self.device = device
            self.model = model
            self._default_model_config = self.model_config
            self._default_model = model

        def set_weight_update_target(self, model, model_config) -> None:
            self.model = model
            self.model_config = model_config

        def reset_weight_update_target(self) -> None:
            self.model = self._default_model
            self.model_config = self._default_model_config

        def parse_init_info(self, init_dict):
            try:
                return self.init_info_cls(**init_dict)
            except TypeError as error:
                raise ValueError(str(error)) from error

        def parse_update_info(self, update_dict):
            try:
                return self.update_info_cls(**update_dict)
            except TypeError as error:
                raise ValueError(str(error)) from error

        def update_weights(self, update_info: dict) -> None:
            self.receive_weights(self.parse_update_info(update_info))

        @abstractmethod
        def init_transfer_engine(self, init_info) -> None:
            raise NotImplementedError

        @abstractmethod
        def start_weight_update(self) -> None:
            raise NotImplementedError

        @abstractmethod
        def finish_weight_update(self) -> None:
            raise NotImplementedError

        @abstractmethod
        def receive_weights(self, update_info) -> None:
            raise NotImplementedError

        @abstractmethod
        def shutdown(self) -> None:
            raise NotImplementedError

        @staticmethod
        @abstractmethod
        def trainer_send_weights(iterator, trainer_args) -> None:
            raise NotImplementedError

    class WeightTransferEngineFactory:
        _registry: dict = {}

        @classmethod
        def register_engine(cls, name, module_path_or_cls, class_name=None):
            if name in cls._registry:
                raise ValueError(
                    f"Weight transfer engine '{name}' is already registered."
                )
            if isinstance(module_path_or_cls, str):
                if class_name is None:
                    raise ValueError(
                        "class_name is required when registering with module path"
                    )
                import importlib

                def loader():
                    module = importlib.import_module(module_path_or_cls)
                    return getattr(module, class_name)

                cls._registry[name] = loader
            else:
                cls._registry[name] = lambda: module_path_or_cls

        @classmethod
        def create_engine(cls, config, vllm_config, device, model):
            if config.backend not in cls._registry:
                raise ValueError(f"Invalid weight transfer backend: {config.backend}.")
            return cls._registry[config.backend]()(config, vllm_config, device, model)

    @dataclass
    class WeightTransferConfig:
        backend: str
        engine_id: str = ""
        rank: int = 0
        local_rank: int = 0
        init_info: dict | None = None

    package = ModuleType("vllm.distributed.weight_transfer")
    base = ModuleType("vllm.distributed.weight_transfer.base")
    setattr(base, "WeightTransferEngine", WeightTransferEngine)
    setattr(base, "WeightTransferInitInfo", WeightTransferInitInfo)
    setattr(base, "WeightTransferUpdateInfo", WeightTransferUpdateInfo)
    factory = ModuleType("vllm.distributed.weight_transfer.factory")
    setattr(factory, "WeightTransferEngineFactory", WeightTransferEngineFactory)
    config = ModuleType("vllm.config.weight_transfer")
    setattr(config, "WeightTransferConfig", WeightTransferConfig)
    sys.modules[package.__name__] = package
    sys.modules[base.__name__] = base
    sys.modules[factory.__name__] = factory
    sys.modules[config.__name__] = config
