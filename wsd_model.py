import torch
from torch import nn
import numpy as np
import os

class WSDModel(nn.Module):
    """
        Model that builds a simple classifier on top of a language model,
        using the output or hidden layers associated with a specified token position
    """
    def __init__(self, base_model, num_labels, logits_mask_fn, use_last_n_layers=1, base_model_name="", classifier_hidden_layers=[], cache_embeddings=False):
        super(WSDModel, self).__init__()
        self.base_model = base_model
        base_output_size = get_base_output_size(base_model, base_model_name)
        if len(classifier_hidden_layers) == 0:
            self.classifier = nn.Linear(base_output_size * use_last_n_layers, num_labels)
        else:
            layer_sizes = [base_output_size * use_last_n_layers] + classifier_hidden_layers
            layers = sum([[nn.Linear(s1,s2), nn.ReLU()] for s1,s2 in zip(layer_sizes,layer_sizes[1:])],[])
            layers += [nn.Linear(layer_sizes[-1], num_labels)]
            self.classifier = nn.Sequential(*layers)
            print("Using classifier " + str(classifier_hidden_layers) + ":")
            print(self.classifier)
            print()
        self.num_labels = num_labels
        self.logits_mask_fn = logits_mask_fn
        self.use_last_n_layers = use_last_n_layers
        self.base_model_name = base_model_name
        self.cache_embeddings = cache_embeddings
        if cache_embeddings:
            self.cached_embeddings_path = self.get_cached_embeddings_path(False)

    def forward(self, x, token_positions=None, lemmas=None, labels=None, example_ids=None):
        """
        :param token_positions: The position of the token we want to query the sense of, for each batch
        """
        features_for_relevant_token = None
        if self.cache_embeddings:
            features_for_relevant_token = self.load_cached_embeddings(example_ids)
        if features_for_relevant_token is None:
            base_model_output = self.base_model(x)
            hidden_states = base_model_output[-1][-self.use_last_n_layers:] # Because we have set config.output_hidden_states=True and config.output_attentions=False
            hidden_states_for_relevant_token = []
            for layer in hidden_states:
                hidden_state_for_relevant_token = layer[list(range(len(token_positions))),token_positions,:]
                hidden_states_for_relevant_token.append(hidden_state_for_relevant_token)
            features_for_relevant_token = torch.cat(hidden_states_for_relevant_token, 1) # Concatenate the last n hidden layers along the neuron dimension
            if self.cache_embeddings:
                self.save_cached_embeddings(features_for_relevant_token, example_ids)

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

    def load_cached_embeddings(self, ids):
        num_embeddings = ids.shape[0]
        loaded_embeddings = []
        for i in range(num_embeddings):
            path = self.cached_embeddings_path + "/" + str(ids[i].item()) + ".npy"
            try:
                loaded_embeddings.append(np.load(path))
            except FileNotFoundError:
                try:
                    emb = np.load("embeddings/" + self.base_model_name + "_last-4" + "/" + str(ids[i].item()) + ".npy")
                    w = emb.shape[1]
                    wanted_w = self.use_last_n_layers * w//4
                    emb = emb[:,-wanted_w:]
                    loaded_embeddings.append(emb)
                except FileNotFoundError:
                    print("*",end="")
                    return None
        return torch.tensor(np.vstack(loaded_embeddings)).cuda()

    def save_cached_embeddings(self, embeddings, ids):
        num_embeddings = ids.shape[0]
        for i in range(num_embeddings):
            path = self.get_cached_embeddings_path(True) + "/" + str(ids[i].item()) + ".npy"
            np.save(path, embeddings[None,i,:].cpu().numpy())

    def get_cached_embeddings_path(self, create_if_not_exists=False):
        path = "embeddings/" + self.base_model_name + "_last-" + str(self.use_last_n_layers)
        if create_if_not_exists and not os.path.exists(path):
            os.makedirs(path)
        return path

    def save_classifier(self, experiment_name, best=False):
        path = "saved_classifiers"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.classifier.state_dict(), path + "/" + experiment_name + ".pt")
        if best:
            torch.save(self.classifier.state_dict(), path+"/"+experiment_name+" [BEST]" + ".pt")

    def load_classifier(self, experiment_name):
        path = "saved_classifiers"
        try:
            self.classifier.load_state_dict(torch.load(path+"/"+experiment_name+".pt"))
            print("Previously found classifier found")
            return True
        except FileNotFoundError:
            print("No previously saved classifier found")
            return False




def get_base_output_size(base_model, base_model_name):
    if base_model_name.startswith('distilbert'):
        return base_model.config.dim
    else:
        return base_model.config.hidden_size



