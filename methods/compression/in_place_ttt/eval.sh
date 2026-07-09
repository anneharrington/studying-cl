#!/bin/bash

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -e


SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

LENGTHS=(4k 8k 16k 32k 64k)
GPU=5

for len in ${LENGTHS[@]}; do
    echo "[$(date)] Starting ruler_${len} on GPU ${GPU}"
    CUDA_VISIBLE_DEVICES=${GPU} python -c "
import inference_model
from opencompass.cli.main import main
import sys
sys.argv = ['opencompass', 'eval_config/ruler_${len}.py', '--debug']
main()
" &
    sleep 5
    GPU=$(( (GPU + 1) % 8 ))
done

echo "[$(date)] All launched. Waiting..."
wait
echo "[$(date)] All done."
