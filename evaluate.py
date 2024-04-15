import os
from time import perf_counter
import numpy as np
import torch
from speed.speed import C9H13N
import torchvision.transforms.functional as TF
from torchvision.io import read_image

# Misc
model_path = 'models/model_epoch_310.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#load the model
model = C9H13N()
model = model.to(device)

model.load_state_dict(torch.load(model_path)['model_state_dict'])
model.eval()

def fscore(pred, true):
    tp = (pred * true).sum()
    fp = (pred * (1 - true)).sum()
    fn = ((1 - pred) * true).sum()
    return tp / (tp + 0.5 * (fp + fn))

images = os.listdir('data/image_2')
fscores = []
times = []

for img in images:
    image_path = os.path.join('data/image_2/', img)
    label_path = os.path.join('data/gt_image_2/', img)

    label = read_image(label_path)
    label = label[2, ...] / 255
    label = label.to(device)

    image = read_image(image_path)
    image = image.to(device)

    image = torch.unsqueeze(image, 0)
    image = TF.normalize(image.type(torch.float), [0, 0, 0], [1, 1, 1])

    with torch.no_grad():
        start = perf_counter()
        output = model(image)
        end = perf_counter()
        time = end - start

    output = torch.argmax(output, axis=1) # Take the argmax to get the class prediction
    output = np.squeeze(output)  # Remove singleton dimensions

    score = fscore(output, label)
    fscores.append(score.item())
    times.append(time)

print("Mean F1 Score:", np.mean(fscores))
print("Mean inference time:", np.mean(times))
