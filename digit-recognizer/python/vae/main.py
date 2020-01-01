import torch
import pandas as pd
from torch import optim

from data import load_data
from model import VAE
from loss import loss_function


# Hyper parameters
log_interval = 100
epochs = 10
batch_size = 10

# Init
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(dev)
optimizer = optim.Adam(model.parameters(), lr=1e-3)


train, test = load_data()

def model_train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train):
        data = data.to(dev)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tloss: {:.0f}'.format(
                epoch, batch_idx * len(data), len(train),
                100. * batch_idx / len(train),
                loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
        epoch, train_loss / len(train)))
                                        

def model_test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data  in enumerate(test):
            data = data.to(dev)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                         recon_batch.view(batch_size, 1, 20, 20)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + str(epoch) + '.png', nrow=n)
    test_loss /= len(test)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, epochs + 1):
    model_train(epoch)
    model_test(epoch)
    with torch.no_grad():
        sample = troch.randn(64, 20).to(dev)
        sample = model.decode(sample).cpu()
        save_imsage(sample.view(64, 1, 28, 28),
                    'results/sample_' + str(epoch) + '.png')



