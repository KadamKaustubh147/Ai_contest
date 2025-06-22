import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# Model class must match the training script
class CVAE(nn.Module):
    def __init__(self):
        super(CVAE, self).__init__()
        self.fc1 = nn.Linear(28*28 + 10, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20 + 10, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def encode(self, x, y):
        h1 = F.relu(self.fc1(torch.cat([x, y], dim=1)))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        h3 = F.relu(self.fc3(torch.cat([z, y], dim=1)))
        return torch.sigmoid(self.fc4(h3))

    def sample(self, y, n_samples=5):
        with torch.no_grad():
            y = torch.eye(10)[y].to(torch.float32)
            y = y.repeat(n_samples, 1)
            z = torch.randn(n_samples, 20)
            samples = self.decode(z, y)
            return samples.view(-1, 28, 28)

model = CVAE()
model.load_state_dict(torch.load("saved_model/cvae_mnist.pt", map_location=torch.device('cpu')))
model.eval()

st.title("MNIST Digit Generator (0–9)")
digit = st.selectbox("Select a digit to generate", list(range(10)))

if st.button("Generate Images"):
    samples = model.sample(torch.tensor(digit), n_samples=5)
    fig, axs = plt.subplots(1, 5, figsize=(10, 2))
    for i in range(5):
        axs[i].imshow(samples[i], cmap='gray')
        axs[i].axis('off')
    st.pyplot(fig)
