from transformers import pipeline

# Load the model
detector = pipeline("image-classification", model="Organika/sdxl-detector")

# Use it
result = detector("./firstImage.png")
print(result[0])