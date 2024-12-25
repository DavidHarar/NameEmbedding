from transformers import BertConfig, BertForMaskedLM, RobertaConfig, RobertaForMaskedLM, DebertaV2Config, DebertaV2ForMaskedLM


def model_factory(model_config: dict):
    """
    Factory function to return a model ready for masked language modeling based on a custom configuration.

    Parameters:
        model_config: model config.

    Returns:
        model: The initialized model configured for masked language modeling.
    """

    model_type = model_config.get('model_type')

    if model_type == "BERT":
        config = BertConfig(**model_config)
        model = BertForMaskedLM(config=config)
    elif model_type == "RoBERTa":
        config = RobertaConfig(**model_config)
        model = RobertaForMaskedLM(config=config)
    elif model_type == "DeBERTa":
        config = DebertaV2Config(**model_config)
        model = DebertaV2ForMaskedLM(config=config)
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Choose from ['BERT', 'RoBERTa', 'DeBERTa'].")

    return model


