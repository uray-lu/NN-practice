from torch import nn

class network(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.model = nn.Sequential(
            #not bais first
            nn.Linear(30,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,32),
            nn.ReLU(inplace=True),
            nn.Linear(32,1),
            
        )

    def forward(self, x):
        output = self.model(x)
        return output
    