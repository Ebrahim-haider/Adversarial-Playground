"""
Adversarial Attack implementations for Trustworthy AI Playground.
Includes:
- FGSM (Fast Gradient Sign Method) - Goodfellow et al.
- PGD (Projected Gradient Descent) - Madry et al.
Supports both untargeted and targeted attacks.
"""

import torch
import torch.nn.functional as F


def fgsm_attack(image: torch.Tensor, epsilon: float, data_grad: torch.Tensor, targeted: bool = False) -> torch.Tensor:
    """
    Generate adversarial sample using Fast Gradient Sign Method (FGSM).
    x_adv = x + epsilon * sign(grad_x J(theta, x, y))  (untargeted)
    x_adv = x - epsilon * sign(grad_x J(theta, x, y_target))  (targeted)
    """
    if epsilon == 0:
        return image.clone()

    # Collect the element-wise sign of the data gradient
    sign_data_grad = data_grad.sign()

    # Create the perturbed image
    if targeted:
        # Move in the direction opposite to the target class loss gradient
        perturbed_image = image - epsilon * sign_data_grad
    else:
        # Move in the direction of increasing true class loss gradient
        perturbed_image = image + epsilon * sign_data_grad

    # Clamp the values to keep them in valid normalized pixel range [0, 1]
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0)
    return perturbed_image


def pgd_attack(model, image: torch.Tensor, target_or_label: torch.Tensor, 
               epsilon: float, alpha: float = 0.01, num_iter: int = 10, 
               targeted: bool = False, device: str = "cpu") -> torch.Tensor:
    """
    Generate adversarial sample using Projected Gradient Descent (PGD / BIM).
    Iteratively applies small FGSM steps and projects back into the L-infinity epsilon ball.
    """
    if epsilon == 0:
        return image.clone()

    original_image = image.clone().detach()
    perturbed_image = image.clone().detach()

    # Optional: start with a small random perturbation within epsilon ball
    perturbed_image = perturbed_image + torch.empty_like(perturbed_image).uniform_(-epsilon, epsilon)
    perturbed_image = torch.clamp(perturbed_image, 0.0, 1.0).detach()

    for _ in range(num_iter):
        perturbed_image.requires_grad = True

        output = model(perturbed_image)
        loss = F.cross_entropy(output, target_or_label)

        model.zero_grad()
        loss.backward()

        data_grad = perturbed_image.grad.data

        if targeted:
            adv_step = perturbed_image - alpha * data_grad.sign()
        else:
            adv_step = perturbed_image + alpha * data_grad.sign()

        # Projection back into the L-infinity epsilon ball around the original image
        eta = torch.clamp(adv_step - original_image, min=-epsilon, max=epsilon)
        perturbed_image = torch.clamp(original_image + eta, min=0.0, max=1.0).detach()

    return perturbed_image
