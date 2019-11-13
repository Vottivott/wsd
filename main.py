import torch

from transformers import AlbertTokenizer
from transformers.modeling_albert import AlbertModel

model_name = "albert-xxlarge-v2"
tokenizer = AlbertTokenizer.from_pretrained(model_name)
model = AlbertModel.from_pretrained(model_name)




with open("results.txt","w") as out:
    s = str(torch.cuda.is_available())
    s += "\n" + str(torch.cuda.get_device_name(torch.cuda.current_device()))
    out.write(s)
