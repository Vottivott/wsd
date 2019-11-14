from wsd import wsd

wsd(model_name='distilbert-base-uncased', classifier_input='token-embedding-last-2-layers')
wsd(model_name='distilbert-base-uncased', classifier_input='token-embedding-last-3-layers')

wsd(model_name='albert-xxlarge-v2', classifier_input='token-embedding-last-2-layers')
wsd(model_name='albert-xxlarge-v2', classifier_input='token-embedding-last-3-layers')
