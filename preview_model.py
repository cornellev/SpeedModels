import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from speed.doublescuffedspeed import C9H13N
import torchvision.transforms.functional as TF
from torchvision.io import read_image


# Load the model
model = C9H13N()
model.load_state_dict(torch.load('models/doublescuffed/best_model.pth')['model_state_dict'])
model.eval()

# Load the image
#image_path = 'data/image_2/110.png'
image_path = 'img.jpg'
image = cv2.imread(image_path)

input_image = read_image(image_path)
input_image = torch.unsqueeze(input_image, 0)  # Add batch dimension
input_image = TF.normalize(input_image.type(torch.float), [0, 0, 0], [1, 1, 1])

# Pass the image through the model
with torch.no_grad():
    output = model(input_image)

# Postprocess the output
output = output.detach().cpu().numpy()  # Convert to numpy array
print(output)
output = np.argmax(output, axis=1)  # Take the argmax to get the class prediction
output = np.squeeze(output)  # Remove singleton dimensions

# Create a blue translucent mask
mask = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
mask[output == 1] = [255, 0, 0]  # Set road pixels to blue

# Overlay the mask on the original image
result = cv2.addWeighted(image, 1, mask, 0.5, 0)  # Apply the mask with 50% transparency

# Display the result
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
cv2.imwrite("image.png", result)
plt.show()