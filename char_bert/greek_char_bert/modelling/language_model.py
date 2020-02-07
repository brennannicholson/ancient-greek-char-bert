# Copyright 2018 The Google AI Language Team Authors,  The HuggingFace Inc. Team and deepset Team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
from farm.modeling.language_model import Bert


class PretrainingBERT(Bert):
    """A BERT model which sets the model attribute correctly, allowing the weights to be initialized for pretraining."""

    def __init__(self, model=None):
        """Allows the model to be initiated from scratch rather than to loading an existing model for finetuning."""
        super(Bert, self).__init__()
        self.model = model
        self.name = "bert"
