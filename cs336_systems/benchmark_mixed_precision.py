import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features:int, out_features:int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        
        x = self.relu(self.fc1(x))
        print ("fc1", x.dtype)
        x = self.ln(x)
        print ("ln", x.dtype)
        x = self.fc2(x)
        print ("logit", x.dtype)
        return x
    
model = ToyModel(10, 2).to("cuda")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Dtype: {param.dtype}")


input = torch.ones(5, 10).to("cuda")
with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    result = model(input)

    target = torch.rand(5, 2).to("cuda")
    loss_fn = nn.MSELoss()
    loss = loss_fn(result, target)
    print ("loss", loss.dtype)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    for name, param in model.named_parameters():
        print(f"gradient of {name}: {param.grad.dtype}")
