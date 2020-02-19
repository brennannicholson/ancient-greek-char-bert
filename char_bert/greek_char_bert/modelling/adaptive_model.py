from farm.modeling.adaptive_model import AdaptiveModel
from char_bert.greek_char_bert.modelling.language_model import PretrainingBERT
from char_bert.greek_char_bert.modelling.prediction_head import CharMLMHead


class CharMLMAdaptiveModel(AdaptiveModel):
    @classmethod
    def load(cls, load_dir, device):
        """
        This method is a copy of AdaptiveModel.load() from farm/modeling/adaptive_model.py. It has been modified to load a PretrainingBERT and a CharMLMHead. - BN

        Loads an AdaptiveModel from a directory. The directory must contain:

        * language_model.bin
        * language_model_config.json
        * prediction_head_X.bin  multiple PH possible
        * prediction_head_X_config.json
        * processor_config.json config for transforming input
        * vocab.txt vocab file for language model, turning text to Wordpiece Tokens

        :param load_dir: location where adaptive model is stored
        :type load_dir: str
        :param device: to which device we want to sent the model, either cpu or cuda
        :type device: torch.device
        """

        # Language Model
        language_model = PretrainingBERT.load(load_dir)

        # Prediction heads
        _, ph_config_files = cls._get_prediction_head_files(load_dir)
        prediction_heads = []
        ph_output_type = []
        for config_file in ph_config_files:
            head = CharMLMHead.load(config_file)
            # set shared weights between LM and PH
            if type(head) == CharMLMHead:
                head.set_shared_weights(
                    language_model.model.embeddings.word_embeddings.weight
                )
            prediction_heads.append(head)
            ph_output_type.append(head.ph_output_type)

        return cls(language_model, prediction_heads, 0.1, ph_output_type, device)
