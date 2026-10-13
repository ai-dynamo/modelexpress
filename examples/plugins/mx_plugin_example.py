# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Loading strategy template: implement all methods before enabling it."""

from modelexpress.load_strategy import (
    LoadContext,
    LoadResult,
    LoadStrategy,
)


class ExampleStrategy(LoadStrategy):
    name = "example"
    requires = ()

    def is_available(self, ctx: LoadContext) -> bool:
        raise NotImplementedError("Implement ExampleStrategy.is_available")

    def load(self, result: LoadResult, ctx: LoadContext) -> LoadResult:
        raise NotImplementedError("Implement ExampleStrategy.load")

    def rollback(self, ctx: LoadContext) -> None:
        raise NotImplementedError("Implement ExampleStrategy.rollback")


def register_plugin() -> type[LoadStrategy]:
    return ExampleStrategy
