import torch

print(torch.cuda.is_available())
with open("results.txt","w") as out:
    s = str(torch.cuda.is_available())
    s += "\n" + str(torch.cuda.get_device_name(torch.cuda.current_device()))
    out.write(s)
