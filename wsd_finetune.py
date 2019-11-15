from wsd import wsd

wsd(classifier_input='token-embedding-last-layer',
    freeze_base_model=False,
    cache_embeddings=False)