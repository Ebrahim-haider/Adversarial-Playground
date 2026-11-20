import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageDraw
import io
import os

from models import get_model, preprocess_image
from attacks import fgsm_attack, pgd_attack
from defenses import (
    apply_gaussian_blur,
    apply_median_filter,
    apply_jpeg_compression,
    apply_bit_depth_reduction
)

st.set_page_config(
    page_title="Trustworthy AI - Adversarial Playground",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for polished look
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #E2E8F0;
        margin-bottom: 10px;
    }
    .badge-success {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-danger {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-warning {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_cached_model(name: str):
    return get_model(name)


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Converts a (1, 3, H, W) tensor in [0, 1] to a PIL Image."""
    np_img = tensor.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    np_img = np.clip(np_img * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def amplify_noise_to_pil(original_tensor: torch.Tensor, perturbed_tensor: torch.Tensor, factor: float = 10.0) -> Image.Image:
    """Amplifies and centers adversarial noise around 128 (gray) for human visualization."""
    diff = (perturbed_tensor - original_tensor).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # Normalize noise into visible dynamic range
    # 0.5 (neutral gray) + scaled perturbation
    vis = 0.5 + diff * factor
    vis = np.clip(vis * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(vis)


def get_top_predictions(model, tensor: torch.Tensor, categories, top_k=3):
    """Returns top_k (class_name, confidence_percentage) pairs."""
    with torch.no_grad():
        output = model(tensor)
        probabilities = F.softmax(output, dim=1)[0]
        top_probs, top_indices = torch.topk(probabilities, top_k)
        
    results = []
    for prob, idx in zip(top_probs, top_indices):
        results.append((categories[idx.item()], prob.item() * 100.0, idx.item()))
    return results


def generate_sample_image(sample_name: str) -> Image.Image:
    """Generates synthetic high-contrast test images if local files aren't found."""
    img = Image.new("RGB", (224, 224), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    if sample_name == "Stop Sign (Simulated)":
        draw.polygon([(60, 20), (164, 20), (210, 66), (210, 158), 
                      (164, 204), (60, 204), (14, 158), (14, 66)], fill="#CC0000")
        draw.rectangle([45, 95, 179, 129], fill="#FFFFFF")
    elif sample_name == "Golden Retriever / Pet (Simulated)":
        draw.ellipse([50, 50, 174, 174], fill="#D4A373")
        draw.ellipse([70, 80, 95, 105], fill="#333333")
        draw.ellipse([129, 80, 154, 105], fill="#333333")
        draw.polygon([(112, 115), (97, 140), (127, 140)], fill="#111111")
    elif sample_name == "Lemon (Simulated)":
        draw.ellipse([40, 60, 184, 164], fill="#FFF017", outline="#D4C800", width=4)
        draw.polygon([(20, 112), (40, 92), (40, 132)], fill="#FFF017")
        draw.polygon([(204, 112), (184, 92), (184, 132)], fill="#FFF017")
    else:
        draw.rectangle([50, 50, 174, 174], fill="#2563EB")
    return img


# ===================== UI Header =====================
st.markdown('<div class="main-title">🛡️ Trustworthy AI: Adversarial Attack & Robustness Playground</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explore the vulnerability of Deep Neural Networks to adversarial perturbations, observe gradient-based attacks in real time, and evaluate input-level defense strategies.</div>', unsafe_allow_html=True)

# ===================== Sidebar Controls =====================
st.sidebar.header("⚙️ Configuration")

# 1. Model Selection
model_name = st.sidebar.selectbox(
    "Choose Vision Model",
    ["ResNet18", "MobileNetV3", "ResNet50"],
    index=0,
    help="Pretrained ImageNet architectures. ResNet18 and MobileNetV3 run very quickly on standard CPU."
)

model, categories = load_cached_model(model_name)

# 2. Input Image Source
st.sidebar.subheader("🖼️ Input Image")
image_source = st.sidebar.radio("Select Image Source:", ["Sample Images", "Upload Image"], index=0)

active_pil_image = None
sample_dir = os.path.join(os.path.dirname(__file__), "sample_images")

if image_source == "Sample Images":
    sample_options = ["Stop Sign", "Giant Panda", "Tabby Cat", "Lemon", "Dog"]
    chosen_sample = st.sidebar.selectbox("Choose a sample:", sample_options)
    
    # Check if image file exists in sample_images directory
    local_path = os.path.join(sample_dir, f"{chosen_sample.lower().replace(' ', '_')}.jpg")
    if os.path.exists(local_path):
        active_pil_image = Image.open(local_path).convert("RGB")
    else:
        active_pil_image = generate_sample_image(f"{chosen_sample} (Simulated)")
else:
    uploaded_file = st.sidebar.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        active_pil_image = Image.open(uploaded_file).convert("RGB")
    else:
        st.info("👈 Please select or upload an image from the sidebar to begin.")
        st.stop()

# 3. Attack Parameters
st.sidebar.subheader("⚡ Adversarial Attack")
attack_type = st.sidebar.selectbox("Attack Algorithm", ["FGSM (Fast Gradient Sign Method)", "PGD (Projected Gradient Descent)"])

attack_mode = st.sidebar.radio("Attack Objective", ["Untargeted (Cause any error)", "Targeted (Force specific class)"])
targeted = (attack_mode.startswith("Targeted"))

target_class_idx = 0
if targeted:
    # A few recognizable ImageNet targets
    target_presets = {
        "Toaster (Class 859)": 859,
        "Lemon (Class 951)": 951,
        "Gibbon (Class 368)": 368,
        "Banana (Class 954)": 954,
        "Speed Limit / Traffic Sign (Class 919)": 919
    }
    target_selection = st.sidebar.selectbox("Target Class", list(target_presets.keys()))
    target_class_idx = target_presets[target_selection]

epsilon = st.sidebar.slider(
    "Perturbation Budget (ε)",
    min_value=0.00,
    max_value=0.25,
    value=0.05,
    step=0.01,
    help="Higher ε adds more noise. Even ε=0.04 (4% pixel change) is often enough to fool the network while remaining virtually invisible to humans."
)

pgd_iters = 10
pgd_alpha = 0.01
if attack_type.startswith("PGD"):
    pgd_iters = st.sidebar.slider("PGD Iterations", min_value=5, max_value=30, value=10, step=1)
    pgd_alpha = st.sidebar.slider("Step Size (α)", min_value=0.005, max_value=0.05, value=0.01, step=0.005)

# 4. Defensive Sanitization
st.sidebar.subheader("🛡️ Defense Mechanism")
enable_defense = st.sidebar.checkbox("Enable Defense Sanitization", value=False)
defense_method = "None"
if enable_defense:
    defense_method = st.sidebar.selectbox(
        "Select Defense Technique",
        ["Gaussian Blur (Spatial Smoothing)", "Median Filter", "JPEG Compression", "Bit-depth Reduction"]
    )
    if defense_method.startswith("Gaussian"):
        blur_kernel = st.sidebar.slider("Blur Kernel Size", min_value=3, max_value=11, value=5, step=2)
        blur_sigma = st.sidebar.slider("Blur Sigma", min_value=0.5, max_value=3.0, value=1.5, step=0.5)
    elif defense_method.startswith("JPEG"):
        jpeg_quality = st.sidebar.slider("JPEG Quality", min_value=10, max_value=80, value=40, step=5)
    elif defense_method.startswith("Bit-depth"):
        bits = st.sidebar.slider("Quantization Bits", min_value=2, max_value=6, value=3, step=1)


# ===================== Inference & Attack Engine =====================

input_tensor = preprocess_image(active_pil_image).to("cpu")
input_tensor.requires_grad = True

# 1. Clean Prediction
clean_preds = get_top_predictions(model, input_tensor, categories, top_k=3)
clean_top_class, clean_top_conf, clean_top_idx = clean_preds[0]

# 2. Compute Attack
if attack_type.startswith("FGSM"):
    output = model(input_tensor)
    if targeted:
        target_tensor = torch.tensor([target_class_idx], dtype=torch.long)
        loss = F.cross_entropy(output, target_tensor)
    else:
        true_label = torch.tensor([clean_top_idx], dtype=torch.long)
        loss = F.cross_entropy(output, true_label)
        
    model.zero_grad()
    loss.backward()
    data_grad = input_tensor.grad.data
    adversarial_tensor = fgsm_attack(input_tensor, epsilon=epsilon, data_grad=data_grad, targeted=targeted)
else: # PGD
    if targeted:
        target_tensor = torch.tensor([target_class_idx], dtype=torch.long)
        adversarial_tensor = pgd_attack(model, input_tensor, target_tensor, epsilon=epsilon, 
                                        alpha=pgd_alpha, num_iter=pgd_iters, targeted=True)
    else:
        true_label = torch.tensor([clean_top_idx], dtype=torch.long)
        adversarial_tensor = pgd_attack(model, input_tensor, true_label, epsilon=epsilon, 
                                        alpha=pgd_alpha, num_iter=pgd_iters, targeted=False)

# 3. Adversarial Prediction
adv_preds = get_top_predictions(model, adversarial_tensor, categories, top_k=3)
adv_top_class, adv_top_conf, adv_top_idx = adv_preds[0]

# 4. Optional Defense Processing
defended_tensor = None
defended_preds = None
if enable_defense:
    if defense_method.startswith("Gaussian"):
        defended_tensor = apply_gaussian_blur(adversarial_tensor, kernel_size=blur_kernel, sigma=blur_sigma)
    elif defense_method.startswith("Median"):
        defended_tensor = apply_median_filter(adversarial_tensor, size=3)
    elif defense_method.startswith("JPEG"):
        defended_tensor = apply_jpeg_compression(adversarial_tensor, quality=jpeg_quality)
    elif defense_method.startswith("Bit-depth"):
        defended_tensor = apply_bit_depth_reduction(adversarial_tensor, bits=bits)
        
    defended_preds = get_top_predictions(model, defended_tensor, categories, top_k=3)


# ===================== Metrics Calculation =====================
diff_tensor = (adversarial_tensor - input_tensor).detach()
l_inf = torch.norm(diff_tensor, p=float('inf')).item()
l_2 = torch.norm(diff_tensor, p=2).item()
mse = torch.mean(diff_tensor ** 2).item()
psnr = 10 * np.log10(1.0 / (mse + 1e-10))

# Status evaluation
if targeted:
    attack_succeeded = (adv_top_idx == target_class_idx)
else:
    attack_succeeded = (adv_top_idx != clean_top_idx)


# ===================== Display Dashboard =====================

num_cols = 4 if enable_defense else 3
cols = st.columns(num_cols)

# Column 1: Clean Original
with cols[0]:
    st.markdown("### 1. Clean Input")
    st.image(active_pil_image, use_container_width=True, caption="Original Unaltered Image")
    st.markdown(f"**Top Prediction:** `{clean_top_class}`")
    st.progress(clean_top_conf / 100.0)
    st.caption(f"Confidence: **{clean_top_conf:.2f}%**")
    
    with st.expander("Top 3 Predictions"):
        for cls_name, conf, _ in clean_preds:
            st.write(f"- **{cls_name}**: {conf:.1f}%")

# Column 2: Amplified Adversarial Perturbation
with cols[1]:
    st.markdown("### 2. Perturbation (Noise)")
    noise_pil = amplify_noise_to_pil(input_tensor, adversarial_tensor, factor=12.0)
    st.image(noise_pil, use_container_width=True, caption=f"Gradient Noise (Amplified ×12, ε={epsilon})")
    
    st.markdown(f"""
    <div class="metric-card">
        <b>Noise Metrics:</b><br>
        • L<sub>∞</sub> Norm: <code>{l_inf:.3f}</code><br>
        • L<sub>2</sub> Norm: <code>{l_2:.2f}</code><br>
        • PSNR: <code>{psnr:.1f} dB</code>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Human eyes cannot spot subtle high-frequency gradients, but deep network activations are heavily skewed by them.")

# Column 3: Adversarial Image
with cols[2]:
    st.markdown("### 3. Adversarial Output")
    adv_pil = tensor_to_pil(adversarial_tensor)
    st.image(adv_pil, use_container_width=True, caption=f"What the Neural Network Sees")
    
    st.markdown(f"**Top Prediction:** `{adv_top_class}`")
    st.progress(adv_top_conf / 100.0)
    st.caption(f"Confidence: **{adv_top_conf:.2f}%**")
    
    if attack_succeeded:
        badge = '<span class="badge-danger">🚨 Attack Succeeded! Model Fooled</span>'
    else:
        badge = '<span class="badge-success">🛡️ Robust: Model Held True Class</span>'
    st.markdown(badge, unsafe_allow_html=True)
    
    with st.expander("Top 3 Adversarial Predictions"):
        for cls_name, conf, _ in adv_preds:
            st.write(f"- **{cls_name}**: {conf:.1f}%")

# Column 4: Defended Image (if enabled)
if enable_defense and defended_tensor is not None:
    with cols[3]:
        st.markdown("### 4. Defended Input")
        defended_pil = tensor_to_pil(defended_tensor)
        st.image(defended_pil, use_container_width=True, caption=f"After {defense_method.split(' ')[0]}")
        
        def_top_class, def_top_conf, def_top_idx = defended_preds[0]
        st.markdown(f"**Top Prediction:** `{def_top_class}`")
        st.progress(def_top_conf / 100.0)
        st.caption(f"Confidence: **{def_top_conf:.2f}%**")
        
        defense_recovered = (def_top_idx == clean_top_idx)
        if defense_recovered:
            def_badge = '<span class="badge-success">✅ Defense Restored True Label</span>'
        else:
            def_badge = '<span class="badge-warning">⚠️ Still Perturbed / Misclassified</span>'
        st.markdown(def_badge, unsafe_allow_html=True)
        
        with st.expander("Top 3 Defended Predictions"):
            for cls_name, conf, _ in defended_preds:
                st.write(f"- **{cls_name}**: {conf:.1f}%")


