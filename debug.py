import torch
import torch.nn as nn
import copy

def param_diff(sd1, sd2):
    if len(sd1.keys()) != len(sd2.keys()):
        raise Exception("Sorry, the model structure of two inputs are different")
    for k in sd1:
        if not k in sd2:
            raise Exception("Sorry, the model structure of two inputs are different")
        else:
            param1 = sd1[k]
            param2 = sd2[k]
            cos_0 = nn.CosineSimilarity(dim=0)
            cos_1 = nn.CosineSimilarity(dim=1)
            print("L2 difference on %s:" % k)
            print("Cosine similarity on dimension 0: %s" % cos_0(param1, param2))
            if param1.ndim > 1:
                print("Cosine similarity on dimension 0: %s" % cos_1(param1, param2))

def ckpt_diff(model1, model2):
    diff = torch.zeros(1)
    for k in model1:
        param1 = model1[k].cpu()
        param2 = model2[k].cpu()
        diff = diff + torch.sum(param1 - param2)
    diff = torch.abs(diff)
    return diff


