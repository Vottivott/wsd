import matplotlib.pyplot as plt
import os

def read_dict_file(fname):
    with open(fname, "r") as f:
        s=f.read()
        start = s.index('{')
        end = s.index('}')
        return eval(s[start:end + 1])

def get_results_collection(folder):
    files = os.listdir(folder)
    return {f.replace(".txt",""):read_dict_file(os.path.join(folder,f)) for f in files}

def plot_results(folder):
    results = get_results_collection(folder)
    sorted_items = sorted(results.items(), key=lambda i: i[1]['val_acc'][-1], reverse=True)
    names = []
    for name,res in sorted_items:
        plt.plot(res['val_acc'])
        final = "%.2f%%" % (100*res['val_acc'][-1])
        names.append(final + " " + name)
    plt.legend(names)
    plt.show()


if __name__ == "__main__":
    plot_results('results')