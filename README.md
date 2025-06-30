# AI Animal Classifier with CIFAR-10 and ResNet18

This is a simple AI demo project that trains an image classification model using the CIFAR-10 dataset and ResNet18 architecture. The trained model is then used to predict the category of an animal image via a simple web UI.

## Features

- Fine-tuned ResNet18 model using CIFAR-10 dataset
- Trained on NVIDIA RTX 4000 Ada GPU
- Flask-based web interface for image URL-based prediction
- GPU-accelerated inference supported

## Project Structure

```
akamai-ai/
├── ai_training.py           # Model training script
├── ai_inference.py          # Inference script with web UI
├── data/                    # CIFAR-10 dataset
├── model/
│   ├── resnet18_cifar10.pth # Trained model weights
│   └── class_names.txt      # Class label names
```

## Animal Categories

Based on CIFAR-10:
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

(Only animal categories are used in the inference UI)

## Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/akamai-gpu-ai-demo.git
cd akamai-gpu-ai-demo

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install torch torchvision flask requests pillow tqdm

# 4. Run training (optional if model already exists)
python ai_training.py

# 5. Run inference web app
python ai_inference.py
```

Then open `http://localhost:5000` in your browser.

## GPU Support

- Training and inference scripts automatically use available CUDA-enabled GPUs.
- Multi-GPU training via `torch.nn.DataParallel` is enabled.

## License

MIT License
