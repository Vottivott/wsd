from wsd import wsd

wsd(model_name='ensemble-distil-2-albert-2')

wsd(model_name='ensemble-distil-2-albert-2', classifier_hidden_layers=[768])
wsd(model_name='ensemble-distil-1-albert-1', classifier_hidden_layers=[768])