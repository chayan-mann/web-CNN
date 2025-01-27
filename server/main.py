from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
# Initialize FastAPI app
app = FastAPI()

# Enable CORS middleware to allow frontend to interact with the backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins or specify only frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the CNN model (your existing code for the model here)
class BrainTumorCNN(torch.nn.Module):
    def __init__(self):
        super(BrainTumorCNN, self).__init__()
        self.conv_layers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),

            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Adjust FC layer based on actual output size from convolution layers
        self.fc_layers = torch.nn.Sequential(
            torch.nn.Linear(128 * 16 * 16, 512),  # Adjust this if needed
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 2)  # Output: 2 classes (Yes/No Tumor)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc_layers(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BrainTumorCNN().to(device)

try:
    model.load_state_dict(torch.load("brain_tumor_cnn.pt", map_location=device, weights_only=True))
    model.eval()  # Set to evaluation mode
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# Define image transformations (Update normalization if needed)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

@app.post("/predict/")
async def predict(file: UploadFile = File(description="Upload an MRI image")):
    try:
        # Read and preprocess image (your existing code for processing the image here)

        # Apply transformations
        image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

        # Model inference (your existing code for making predictions here)
        with torch.no_grad():
            output = model(image)
            predicted_class = torch.argmax(output, dim=1).item()

        # Define class labels
        label_map = {0: "No Tumor", 1: "Tumor Detected"}
        return JSONResponse(content={"prediction": label_map[predicted_class]})
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

# Run API with: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)