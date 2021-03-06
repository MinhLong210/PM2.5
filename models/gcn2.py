# input (batch,nodes,features)
# adj (batch,nodes,nodes)
import torch
import torch.nn as nn
class GCN(nn.Module):
  def __init__(self,in_feas,out_feas,hidden_unit):
    super(GCN,self).__init__()
    self.linear1 = nn.Linear(in_feas,hidden_unit,bias = True)
    self.relu1 = nn.ReLU()
    self.linear2 = nn.Linear(hidden_unit,out_feas,bias = True)
    self.sofmax = nn.Softmax(out_feas)


  def weight_inti(self,m):
    if isinstance(m,nn.Linear):
      torch.nn.init.xavier_uniform_(m.weight.data)
      if m.bias is not None:
        m.bias.data.fill_(0.0)


  def forward(self,x,adj):
    x = torch.bmm(adj,x)
    x = self.linear1(x)
    x = self.relu1(x)
    x = torch.bmm(adj,x)
    x = self.linear2(x)
    return self.sofmax(x)