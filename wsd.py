import sys, os
import time
from collections import defaultdict
import random
import torch
from torchtext.data import Field, LabelField, TabularDataset, Iterator, BucketIterator, Example, Dataset
from transformers import AdamW, DistilBertTokenizer, DistilBertModel
import numpy as np

from wsd_model import WSDModel

def wsd(model_name='distilbert-base-uncased',
        classifier_input='token-embedding-last-4-layers', # token-embedding-last-layer / token-embedding-last-n-layers
        classifier_hidden_layers=[],
        reduce_options=True,
        freeze_base_model=True,
        max_len=512,
        batch_size=32,
        save_embeddings=True, # If true, instead of training the program generates embeddings for all training, validation and test examples. Also saves labels
        load_embeddings=False, # If true, preembedded batches are used instead of running the base mode. Also uses saved labels
        test=False,
        lr=5e-5,
        eps=1e-8,
        n_epochs=100):
    train_path = "wsd_train.txt"
    test_path = "wsd_test_blind.txt"
    n_classes = 222
    device = 'cuda'

    import __main__ as main
    print("Script: " + os.path.basename(main.__file__))


    print("Loading base model %s..." % model_name)
    if model_name.startswith('distilbert'):
        tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        if not load_embeddings:
            base_model = DistilBertModel.from_pretrained(model_name, num_labels=n_classes, output_hidden_states=True, output_attentions=False)
    elif model_name.startswith('bert'):
        from transformers import BertTokenizer, BertModel
        tokenizer = BertTokenizer.from_pretrained(model_name)
        if not load_embeddings:
            base_model = BertModel.from_pretrained(model_name, num_labels=n_classes, output_hidden_states=True, output_attentions=False)
    elif model_name.startswith('albert'):
        from transformers import AlbertTokenizer
        from transformers.modeling_albert import AlbertModel
        tokenizer = AlbertTokenizer.from_pretrained(model_name)
        if not load_embeddings:
            base_model = AlbertModel.from_pretrained(model_name, output_hidden_states=True, output_attentions=False)

    use_n_last_layers = 1
    if classifier_input == 'token-embedding-last-layer':
        use_n_last_layers = 1
    elif classifier_input.startswith('token-embedding-last-') and classifier_input.endswith('-layers'):
        use_n_last_layers = int(classifier_input.replace('token-embedding-last-',"").replace('-layers',""))
    else:
        raise ValueError("Invalid classifier_input argument")
    print("Using the last %d layers" % use_n_last_layers)

    def tokenize(str):
        return tokenizer.tokenize(str)[:max_len-2]

    SENSE = LabelField(is_target=True)
    LEMMA = LabelField()
    TOKEN_POS = LabelField(use_vocab=False)
    TEXT = Field(tokenize=tokenize, pad_token=tokenizer.pad_token, init_token=tokenizer.cls_token,
                 eos_token=tokenizer.sep_token)
    INDEX = LabelField(use_vocab=False)
    fields = [('sense', SENSE),
              ('lemma', LEMMA),
              ('token_pos', TOKEN_POS),
              ('text', TEXT),
              ('index', INDEX)]

    def read_data(corpus_file, fields, max_len=None):
        with open(corpus_file, encoding='utf-8') as f:
            examples = []
            for i,line in enumerate(f):
                sense, lemma, word_position, text = line.split('\t')
                if load_embeddings:
                    text = "" # We don't need the text if we load the embeddings
                # We need to convert from the word position to the token position
                words = text.split()
                pre_word = " ".join(words[:int(word_position)])
                pre_word_tokenized = tokenizer.tokenize(pre_word)
                token_position = len(pre_word_tokenized) + 1  # taking into account the later addition of the start token
                if max_len is None or token_position < max_len-1: # ignore examples where the relevant token is cut off due to max_len
                    if sense == '?':
                        sense = 'professional%3:01:00::' # we use this as a placeholder in the test case
                    examples.append(Example.fromlist([sense, lemma, token_position, text, i], fields))
        return Dataset(examples, fields)

    dataset = read_data(train_path, fields, max_len)
    print("First 2 examples:")
    print(dataset[0].sense)
    print(dataset[1].sense)

    # trn, vld = dataset.split(0.7, stratified=True, strata_field='sense')
    trn_indices = np.load("trn_indices.npy")
    vld_indices = np.load("vld_indices.npy")
    trn = Dataset([dataset[i] for i in trn_indices], fields)
    vld = Dataset([dataset[i] for i in vld_indices], fields)
    print("First 5 trn:")
    print(list(map(lambda x: x.sense, trn[:5])))
    print("First 5 vld:")
    print(list(map(lambda x: x.sense, vld[:5])))

    TEXT.build_vocab([])
    if model_name.startswith('albert'):
        class Mapping:
            def __init__(self, fn):
                self.fn = fn
            def __getitem__(self, item):
                return self.fn(item)
        TEXT.vocab.stoi = Mapping(tokenizer.sp_model.PieceToId)
        TEXT.vocab.itos = Mapping(tokenizer.sp_model.IdToPiece)
    else:
        TEXT.vocab.stoi = tokenizer.vocab
        TEXT.vocab.itos = list(tokenizer.vocab)
    SENSE.build_vocab(trn)
    LEMMA.build_vocab(trn)

    if save_embeddings:
        model = WSDModel(base_model, n_classes, lambda a,b:a, use_n_last_layers, model_name, classifier_hidden_layers)
        model.cuda()
        trn_iter = Iterator(trn, device=device, batch_size=batch_size, sort=False, sort_within_batch=False,
                            repeat=False, train=False)
        vld_iter = Iterator(vld, device=device, batch_size=batch_size, sort=False, sort_within_batch=False,
                            repeat=False, train=False)
        tst = read_data(test_path, fields, max_len=512)
        tst_iter = Iterator(tst, device=device, batch_size=batch_size, sort=False, sort_within_batch=False,
                            repeat=False, train=False)
        iters = [('trn',trn_iter),('vld',vld_iter),('tst',tst_iter)]
        model.eval()
        for name,iter in iters:
            print("Saving %s embeddings for %s..." % (name,model_name))
            for i,batch in enumerate(iter):
                if i<10:
                    print(SENSE.vocab.itos[batch.sense[0]])
                print('.', end='')
                sys.stdout.flush()
                text = batch.text.t()
                with torch.no_grad():
                    batch_features = model(text, token_positions=batch.token_pos, lemmas=batch.lemma, labels=batch.sense, save_embeddings=True)
                    batch_features = batch_features.cpu().numpy()
                    directory = "embeddings/" + model_name + " " + classifier_input + "/" + name
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    np.save(directory + "/" + str(i) + ".npy", batch_features)
                    directory_labels = "embeddings/" + model_name + " " + classifier_input + "/" + name + " labels"
                    if not os.path.exists(directory_labels):
                        os.makedirs(directory_labels)
                    np.save(directory_labels + "/" + str(i) + ".npy", batch.sense.cpu().numpy())
                    directory_lemmas = "embeddings/" + model_name + " " + classifier_input + "/" + name + " lemmas"
                    if not os.path.exists(directory_lemmas):
                        os.makedirs(directory_lemmas)
                    np.save(directory_lemmas + "/" + str(i) + ".npy", batch.lemma.cpu().numpy())
        print("Finished saving embeddings!")
        exit(0)
    else:
        if load_embeddings:
            class EmbIter:
                def __init__(self, dataset_name):
                    self.dataset_name = dataset_name
                    self.num_batches = get_num_batches(model_name,classifier_input,self.dataset_name)

                def __iter__(self):
                    for i in range(self.num_batches):
                        yield load_embedding_batch(i,model_name,classifier_input,self.dataset_name)
            trn_iter = EmbIter('trn')
            vld_iter = EmbIter('vld')
        else:
            trn_iter = BucketIterator(trn, device=device, batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, train=True, sort=True)
            vld_iter = BucketIterator(vld, device=device, batch_size=batch_size, sort_key=lambda x: len(x.text), repeat=False, train=False, sort=True)

        if not load_embeddings:
            if freeze_base_model:
                for mat in base_model.parameters():
                    mat.requires_grad = False # Freeze Bert model so that we only train the classifier on top

        if reduce_options:
            lemma_mask = defaultdict(lambda: torch.zeros(len(SENSE.vocab), device=device))
            for example in trn:
                lemma = LEMMA.vocab.stoi[example.lemma]
                sense = SENSE.vocab.stoi[example.sense]
                lemma_mask[lemma][sense] = 1
            lemma_mask = dict(lemma_mask)

            def mask(batch_logits, batch_lemmas): # Masks out the senses that do not belong to the specified lemma
                for batch_i in range(len(batch_logits)):
                    lemma = batch_lemmas[batch_i].item()
                    batch_logits[batch_i, :] *= lemma_mask[lemma]
                return batch_logits
        else:
            def mask(batch_logits, batch_lemmas):
                return batch_logits

        if load_embeddings:
            emb_size = get_embedding_size(model_name, classifier_input, "trn")
            model = WSDModel(emb_size, n_classes, mask, use_n_last_layers, model_name, classifier_hidden_layers)
        else:
            model = WSDModel(base_model, n_classes, mask, use_n_last_layers, model_name, classifier_hidden_layers)

        model.cuda()

        experiment_name = model_name + " " + classifier_input + " " + str(classifier_hidden_layers) + " (" +  (" reduce_options" if reduce_options else "") + (" freeze_base_model" if reduce_options else "") + "  ) " + "max_len=" + str(max_len) + " batch_size=" + str(batch_size) + " lr="+str(lr) + " eps="+str(eps) + (" load_embeddings" if load_embeddings else "")
        print("Starting experiment  " + experiment_name)
        if test:
            tst = read_data(test_path, fields, max_len=512)
            tst_iter = Iterator(tst, device=device, batch_size=batch_size, sort=False, sort_within_batch=False, repeat=False, train=False)
            batch_predictions = []
            for batch in tst_iter:
                print('.', end='')
                sys.stdout.flush()
                text = batch.text.t()
                with torch.no_grad():
                    outputs = model(text, token_positions=batch.token_pos, lemmas=batch.lemma)
                    scores = outputs[-1]
                batch_predictions.append(scores.argmax(dim=1))
            batch_preds = torch.cat(batch_predictions, 0).tolist()
            predicted_senses = [SENSE.vocab.itos(pred) for pred in batch_preds]
            with open("test_predictions/"+experiment_name+".txt", "w") as out:
                out.write("\n".join(predicted_senses))
        else:
            no_decay = ['bias', 'LayerNorm.weight']
            decay = 0.01
            optimizer_grouped_parameters = [
                {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                 'weight_decay': decay},
                {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            optimizer = AdamW(optimizer_grouped_parameters, lr=lr, eps=eps)
            def save_results(history):
                with open("results/" + experiment_name + ".txt", "w") as out:
                    out.write(str(history))

            train(model, optimizer, trn_iter, vld_iter, n_epochs, save_results, load_embeddings)

def train(model, optimizer, trn_iter, vld_iter, n_epochs, epoch_callback=None, load_embeddings=False):
    def evaluate_validation(scores, gold):
        guesses = scores.argmax(dim=1)
        return (guesses == gold).sum().item()

    history = defaultdict(list)
    for i in range(n_epochs):
        t0 = time.time()
        loss_sum = 0
        n_batches = 0
        model.train()
        for batch in trn_iter:
            print('.', end='')
            sys.stdout.flush()
            if load_embeddings:
                emb, labels, lemmas = batch
                outputs = model(emb, lemmas=lemmas, labels=labels)
            else:
                text = batch.text.t()
                optimizer.zero_grad()
                outputs = model(text, token_positions=batch.token_pos, lemmas=batch.lemma, labels=batch.sense)
            loss = outputs[0]

            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            n_batches += 1

            if n_batches % 50 == 0:
                print(f' ({loss_sum / n_batches:.4f})')

        train_loss = loss_sum / n_batches
        history['train_loss'].append(train_loss)
        print(f' ({train_loss:.4f})')
        n_correct = 0
        n_valid = len(vld_iter.dataset)
        loss_sum = 0
        n_batches = 0

        model.eval()
        for batch in vld_iter:
            print('.', end='')
            sys.stdout.flush()
            if load_embeddings:
                emb, labels, lemmas = batch
                with torch.no_grad():
                    outputs = model(emb, lemmas=lemmas, labels=labels)
                    loss_batch, scores = outputs
            else:
                text = batch.text.t()
                with torch.no_grad():
                    outputs = model(text, token_positions=batch.token_pos, lemmas=batch.lemma, labels=batch.sense)
                    loss_batch, scores = outputs

            loss_sum += loss_batch.item()
            n_correct += evaluate_validation(scores, batch.sense)
            n_batches += 1
            if n_batches % 50 == 0:
                print()

        val_acc = n_correct / n_valid
        val_loss = loss_sum / n_batches

        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        t1 = time.time()
        print()
        print(
            f'Epoch {i + 1}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}, val  acc: {val_acc:.4f}, time = {t1 - t0:.4f}')
        if epoch_callback is not None:
            epoch_callback(history)

def load_embedding_batch(index, model_name, classifier_input, dataset_name):
    emb_path = "embeddings/" + model_name + " " + classifier_input + "/" + dataset_name
    labels_path = emb_path + " labels"
    lemmas_path = emb_path + " lemmas"
    emb = np.load(emb_path + "/" + str(index) + ".npy")
    labels = np.load(labels_path + "/" + str(index) + ".npy")
    lemmas = np.load(lemmas_path + "/" + str(index) + ".npy")
    emb, labels, lemmas = torch.tensor(emb).cuda(),torch.tensor(labels).cuda(),torch.tensor(lemmas).cuda()
    return emb, labels, lemmas

def get_num_batches(model_name, classifier_input, dataset_name):
    emb_path = "embeddings/" + model_name + " " + classifier_input + "/" + dataset_name
    return len(os.listdir(emb_path))

def get_embedding_size(model_name, classifier_input, dataset_name):
    emb_path = "embeddings/" + model_name + " " + classifier_input + "/" + dataset_name
    emb = np.load(emb_path + "/" + str(0) + ".npy")
    return emb.shape[1]

if __name__ == "__main__":
    wsd()
