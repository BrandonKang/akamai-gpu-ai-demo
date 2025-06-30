from flask import Flask, request, render_template_string
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO
from collections import OrderedDict

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model structure
model = models.resnet18(weights=None, num_classes=10)

# Load model weights with "module." prefix handled
state_dict = torch.load("model/resnet18_cifar10.pth", map_location=device)
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("module.", "")  # remove 'module.' prefix if exists
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

# Load class names
with open("model/class_names.txt", "r") as f:
    class_names = [line.strip() for line in f.readlines()]

# Define image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Flask Web UI
app = Flask(__name__)

HTML_TEMPLATE = '''
<!doctype html>
<title>Animal Classifier</title>
<h2>Enter Image URL to Predict Animal Category</h2>
<form method=post>
  <input type=text name=img_url size=100>
  <input type=submit value=Predict>
</form>
{% if prediction %}
  <h3>Prediction: {{ prediction }}</h3>
  <img src="{{ image_url }}" width="224">
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_url = None

    if request.method == 'POST':
        image_url = request.form['img_url']
        try:
            response = requests.get(image_url)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs, 1)
                prediction = class_names[predicted.item()]
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(HTML_TEMPLATE, prediction=prediction, image_url=image_url)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

