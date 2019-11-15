from wsd import wsd

wsd(classifier_input='token-embedding-last-layer',
    cls_token=True,
    cache_embeddings=False)