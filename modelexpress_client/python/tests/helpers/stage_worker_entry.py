# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stage worker entry point for tests: object storage is a directory.

The worker builds its own generator client, so a test cannot hand it a reader.
This swaps the S3 client for the file-backed reader, as the tests do in their
own process, and then runs the real worker.
"""

import os
import sys
import time

from modelexpress_rl.inference.methods import canonical_delta
from modelexpress_rl.inference.stage_worker import main
from tests.helpers import refit_stage_process as fakes


def _reader(**_kwargs):
    return fakes.FileReader(os.environ[fakes.READER_ROOT_ENV])


if __name__ == "__main__":
    if os.environ.get(fakes.STALL_ENV):
        time.sleep(60.0)
    canonical_delta.S3Client = _reader
    sys.exit(main())
