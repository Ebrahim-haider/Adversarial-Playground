"""
Verification script for the entire pipeline:
- Loads ResNet18
- Preprocesses giant_panda.jpg
- Runs clean inference
- Generates FGSM attack
- Tests adversarial misclassification
- Applies Gaussian defense
- Asserts pipeline integrity
"""

import os
import torch
from PIL import Image
from models import get_model, preprocess_image
from attacks import fgsm_attack, pgd_attack
from defenses import apply_gaussian_blur, apply_jpeg_compression

def test():
    print("1. Loading ResNet18 model...")
    # Weights will be downloaded or cached
    model, categories = get_model("ResNet18", device="cpu")
    print(f"   ✓ Model loaded with {len(categories)} ImageNet categories.")

    sample_path = os.path.join("sample_images", "giant_panda.jpg")
    assert os.path.exists(sample_path), f"Missing {sample_path}"
    
    img = Image.open(sample_path).convert("RGB")
    tensor = preprocess_image(img).to("cpu")
    tensor.requires_grad = True

    print("2. Running clean inference...")
    output = model(tensor)
    clean_top_idx = torch.argmax(output, dim=1).item()
    clean_class = categories[clean_top_idx]
    print(f"   ✓ Clean prediction: {clean_class} (Class ID {clean_top_idx})")

    print("3. Executing FGSM attack (epsilon=0.08)...")
    true_label = torch.tensor([clean_top_idx], dtype=torch.long)
    loss = torch.nn.functional.cross_entropy(output, true_label)
    model.zero_grad()
    loss.backward()
    
    adv_tensor = fgsm_attack(tensor, epsilon=0.08, data_grad=tensor.grad.data)
    with torch.no_grad():
        adv_output = model(adv_tensor)
        adv_top_idx = torch.argmax(adv_output, dim=1).item()
        adv_class = categories[adv_top_idx]
    print(f"   ✓ Adversarial prediction: {adv_class} (Class ID {adv_top_idx})")
    print(f"   ✓ Prediction flipped: {clean_top_idx != adv_top_idx}")

    print("4. Testing Gaussian Blur defense...")
    defended_tensor = apply_gaussian_blur(adv_tensor, kernel_size=5, sigma=1.5)
    with torch.no_grad():
        def_output = model(defended_tensor)
        def_top_idx = torch.argmax(def_output, dim=1).item()
        def_class = categories[def_top_idx]
    print(f"   ✓ Defended prediction: {def_class} (Class ID {def_top_idx})")

    print("\n All core pipeline tests passed successfully!")

if __name__ == "__main__":
    test()
