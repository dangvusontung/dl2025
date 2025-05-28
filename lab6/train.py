from PIL import Image
from components.cnn_model import SimpleCNN

def load_image_as_2d_array(path):
    img = Image.open(path).convert("L")  # convert to grayscale
    img = img.resize((28, 28))
    pixels = list(img.getdata())
    pixels = [p / 255.0 for p in pixels]
    return [pixels[i * 28:(i + 1) * 28] for i in range(28)]

def main():
    image_path = "sample_digit.png"  # ⚠️ You need to provide this file manually
    image = load_image_as_2d_array(image_path)

    model = SimpleCNN()
    output = model.forward(image)

    print("Prediction output:")
    for i, val in enumerate(output):
        print(f"Class {i}: {val:.4f}")

if __name__ == "__main__":
    main()
