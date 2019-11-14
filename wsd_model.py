import torch
from torch import nn


class WSDModel(nn.Module):
    """
        Model that builds a simple classifier on top of a language model,
        using the output or hidden layers associated with a specified token position
    """
    def __init__(self, base_model, num_labels, logits_mask_fn, use_n_last_layers=1, base_model_name="", classifier_hidden_layers=[]):
        super(WSDModel, self).__init__()
        self.base_model = base_model
        base_output_size = get_base_output_size(base_model, base_model_name)
        if len(classifier_hidden_layers) == 0:
            self.classifier = nn.Linear(base_output_size * use_n_last_layers, num_labels)
        else:
            layer_sizes = [base_output_size * use_n_last_layers] + classifier_hidden_layers
            layers = sum([[nn.Linear(s1,s2), nn.ReLU()] for s1,s2 in zip(layer_sizes,layer_sizes[1:])],[])
            layers += [nn.Linear(layer_sizes[-1], num_labels)]
            self.classifier = nn.Sequential(*layers)
            print("Using classifier " + str(classifier_hidden_layers) + ":")
            print(self.classifier)
            print()
        self.num_labels = num_labels
        self.logits_mask_fn = logits_mask_fn
        self.use_n_last_layers = 1
        self.base_model_name = base_model_name

    def forward(self, x, token_positions=None, lemmas=None, labels=None):
        """
        :param token_positions: The position of the token we want to query the sense of, for each batch
        """
        base_model_output = self.base_model(x)
        hidden_states = base_model_output[-1][-self.use_n_last_layers:] # Because we have set config.output_hidden_states=True and config.output_attentions=False
        hidden_states_for_relevant_token = []
        for layer in hidden_states:
            hidden_state_for_relevant_token = layer[list(range(len(token_positions))),token_positions,:]
            hidden_states_for_relevant_token.append(hidden_state_for_relevant_token)
        features_for_relevant_token = torch.cat(hidden_states_for_relevant_token, 1) # Concatenate the last n hidden layers along the neuron dimension

        logits = self.classifier(features_for_relevant_token)
        logits = self.logits_mask_fn(logits, lemmas)
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits


def get_base_output_size(base_model, base_model_name):
    if base_model_name.startswith('distilbert'):
        return base_model.config.dim
    else:
        return base_model.config.hidden_size
