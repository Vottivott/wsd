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

def plot_results(folder, required_keywords=None, label_fn=None, styles=None, max_epochs=50):
    results = get_results_collection(folder)
    if required_keywords is not None:
        results = {name:res for name,res in results.items() if any(keyword in name for keyword in required_keywords)}
    sorted_items = sorted(results.items(), key=lambda i: max(i[1]['val_acc'][:max_epochs]), reverse=True)
    names = []
    for i,(name,res) in enumerate(sorted_items):
        if styles is not None and i < len(styles):
            plt.plot(range(len(res['val_acc'][:max_epochs])), res['val_acc'][:max_epochs], styles[i])
        else:
            plt.plot(res['val_acc'][:max_epochs])
        final = "%.2f%%" % (100*max(res['val_acc'][:max_epochs]))
        if label_fn is not None:
            name = label_fn(name)
        names.append(final + " " + name)
    plt.xlabel("Epoch")
    plt.ylabel("Validation accuracy")
    plt.legend(names)
    plt.show()



if __name__ == "__main__":
    plot_results('results')