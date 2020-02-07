import os

from farm.infer import Inferencer
from farm.utils import initialize_device_settings
from greek_char_bert.modelling.adaptive_model import CharMLMAdaptiveModel
from greek_char_bert.data_handler.processor import CharMLMProcessor


class CharMLMInferencer(Inferencer):
    """An inference for use with CharMLM components."""

    @classmethod
    def load(cls, load_dir, batch_size=4, gpu=True):
        """
        This method is a copy of Inferencer.load() from farm/infer.py. It has been modified to load CharMLM versions of the AdaptiveModel and Processor. - BN

        Original documentation:

        Initializes inferencer from directory with saved model.
        :param load_dir: Directory where the saved model is located.
        :type load_dir: str
        :param batch_size: Number of samples computed once per batch
        :type batch_size: int
        :param gpu: If GPU shall be used
        :type gpu: bool
        :return: An instance of the Inferencer.
        """

        device, n_gpu = initialize_device_settings(
            use_cuda=gpu, local_rank=-1, fp16=False
        )

        model = CharMLMAdaptiveModel.load(load_dir, device)
        processor = CharMLMProcessor.load_from_dir(load_dir)
        name = os.path.basename(load_dir)
        return cls(model, processor, batch_size=batch_size, gpu=gpu, name=name)
