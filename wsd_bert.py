from wsd import wsd

#wsd(model_name='bert-base-uncased')
wsd(model_name='bert-base-uncased',
        classifier_input='token-embedding-last-layer')
#wsd(model_name='bert-base-uncased',
#        classifier_input='token-embedding-last-2-layers')
wsd(model_name='bert-base-uncased',
        classifier_input='token-embedding-last-layer',classifier_hidden_layers=[768])
#wsd(model_name='bert-base-uncased',
#        classifier_input='token-embedding-last-2-layers',classifier_hidden_layers=[768])
