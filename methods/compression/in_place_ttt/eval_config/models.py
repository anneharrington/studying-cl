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

from opencompass.models import HuggingFaceBaseModel
from opencompass.utils.text_postprocessors import extract_non_reasoning_content

_model_configs = [
    ("your_model", "/path/to/your_hf_model"),
    # (model_name,path)
]

models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr=name,
        path=path,
        engine_config=dict(session_len=131072, max_batch_size=200, tp=1),
        gen_config=dict(
            top_k=20,
            temperature=0,
            top_p=0.95,
            do_sample=False,
            enable_thinking=False,
            max_new_tokens=64,
        ),
        max_seq_len=131072,
        max_out_len=64,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        pred_postprocessor=dict(type=extract_non_reasoning_content),
    )
    for name, path in _model_configs
]
