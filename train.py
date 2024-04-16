import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn.functional as F
from speed.triplescuffedspeed import C9H13N
from kittiset import KittiTraining

def train_one_epoch(model, dataloader, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        labels = labels.long()
        #weight = torch.tensor([2, 1], dtype=torch.float).to(device)
        loss = F.cross_entropy(outputs, labels)
        #loss = weighted_bce_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def save_model(model, optimizer, epoch, loss, path ):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def log_metrics(epoch, loss):
    print(f"Epoch: {epoch}, Loss: {loss:.4f}")


def train_model(epochs, model, training_loader, optimizer, device, resume_path=None):
    writer = SummaryWriter('runs/experiment')
    start_epoch = 0
    best_loss = float('inf')

    model.to(device)

    if resume_path is not None:
        checkpoint = torch.load(resume_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['loss']
        print(f"Resumed training from epoch {start_epoch}, with loss of {best_loss:.4f}")

    for epoch in range(start_epoch, epochs):
        loss = train_one_epoch(model, training_loader, optimizer, device)

        writer.add_scalar('Loss/train', loss, epoch)

        log_metrics(epoch, loss)

        # Save the model if it has a better performance or every 10 epochs
        if loss < best_loss:
            best_loss = loss
            save_model(model, optimizer, epoch, loss, 'models/triplescuffed/best_model.pth')
            print("Saved new best model")

        if epoch % 10 == 0:
            save_path = os.path.join('./models', f'model_epoch_{epoch}.pth')
            save_model(model, optimizer, epoch, loss, save_path)
            print(f"Model checkpoint saved to {save_path}")
    writer.close()
    torch.cuda.empty_cache()


device = torch.device("cuda")

training_set = KittiTraining()
training_loader = torch.utils.data.DataLoader(training_set, batch_size=32, shuffle=True)

model = C9H13N()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
#optimizer = torch.optim.SGD(model.parameters(), lr=3e-4, momentum=0.9, weight_decay=0.0005)

resume_path = None  # e.g., './models/first_try.pth'
train_model(1000, model, training_loader, optimizer, device, resume_path)
