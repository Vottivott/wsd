import torch
from torch import nn
import numpy as np
import os

class WSDEnsembleModel(nn.Module):
    """
        Model that builds a simple classifier on top of the combined (precomputed) embeddings of multiple language models
    """
    def __init__(self, use_n_last_distil, use_n_last_albert, num_labels, logits_mask_fn, classifier_hidden_layers=[]):
        super(WSDEnsembleModel, self).__init__()
        base_output_size = use_n_last_distil*768 + use_n_last_albert*4096
        if len(classifier_hidden_layers) == 0:
            self.classifier = nn.Linear(base_output_size, num_labels)
        else:
            layer_sizes = [base_output_size] + classifier_hidden_layers
            layers = sum([[nn.Linear(s1,s2), nn.ReLU()] for s1,s2 in zip(layer_sizes,layer_sizes[1:])],[])
            layers += [nn.Linear(layer_sizes[-1], num_labels)]
            self.classifier = nn.Sequential(*layers)
            print("Using classifier " + str(classifier_hidden_layers) + ":")
            print(self.classifier)
            print()
        self.num_labels = num_labels
        self.logits_mask_fn = logits_mask_fn
        self.use_n_last_distil = use_n_last_distil
        self.use_n_last_albert = use_n_last_albert

    def forward(self, x, token_positions=None, lemmas=None, labels=None, example_ids=None):
        """
        :param token_positions: The position of the token we want to query the sense of, for each batch
        """
        features_distil = self.load_cached_embeddings(example_ids, 'distilbert-base-uncased', self.use_n_last_distil)
        features_albert = self.load_cached_embeddings(example_ids, 'albert-xxlarge-v2', self.use_n_last_albert)
        features_for_relevant_token = torch.cat((features_distil, features_albert),1)

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

    def load_cached_embeddings(self, ids, name, use_last_n_layers):
        num_embeddings = ids.shape[0]
        loaded_embeddings = []
        for i in range(num_embeddings):
            try:
                emb = np.load("embeddings/" + name + "_last-4" + "/" + str(ids[i].item()) + ".npy")
                w = emb.shape[1]
                wanted_w = use_last_n_layers * w//4
                emb = emb[:,-wanted_w:]
                loaded_embeddings.append(emb)
            except FileNotFoundError:
                print("*",end="")
                return None
        return torch.tensor(np.vstack(loaded_embeddings)).cuda()

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