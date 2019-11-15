from results_plotter import plot_results
import matplotlib as plt

def dict_based_name(d, name):
    for k,v in d.items():
        if k in name:
            return v

def model_name(name):
    d = {'distilbert-base-uncased': 'DistilBERT',
         'bert-base-uncased': 'BERT',
         'albert-xxlarge-v2': 'ALBERT',
         'ensemble-distil-1-albert-1': 'ensemble-last-1',
         'ensemble-distil-2-albert-2': 'ensemble-last-2',
         'ensemble-distil-3-albert-3': 'ensemble-last-3',
         'ensemble-distil-4-albert-4': 'ensemble-last-4'}
    n = dict_based_name(d, name)
    if not name.startswith('ensemble'):
        d = {'token-embedding-last-2':'-last-2',
             'token-embedding-last-3':'-last-3',
             'token-embedding-last-4':'-last-4',
             'token-embedding-last-1':'-last-1',
             'token-embedding-last-layer':'-last-1'}
        return n + dict_based_name(d,name)
    return n

def linear(name):
    if '[768]' in name:
        return 'non-linear classifier'
    else:
        return 'linear classifier'

def cls_token(name):
    if 'cls_token' in name:
        return "cls token"
    else:
        return "relevant token"


def plot(fig_index):
    if fig_index==0:
        #plt.title("Linear vs two-layer classifier")
        #plot_results("results_lin", None, lambda s: model_name(s) + " " + linear(s), ['r-', 'g-', 'r--', 'g--','b-','b--'])#,'c-','c--'])
        plot_results("results_lin", None, lambda s: model_name(s) + " " + linear(s), ['r-', 'r--','b-','b--'])#,'c-','c--'])
    elif fig_index==1:
        plot_results("results_positional", None, lambda s: model_name(s) + " " + cls_token(s), ['r-', 'r--'])#,'c-','c--'])
    elif fig_index==2:
        plot_results("results_gru", None, lambda s: s, ['r-', 'r--'])#,'c-','c--'])


if __name__ == "__main__":
    plot(2)