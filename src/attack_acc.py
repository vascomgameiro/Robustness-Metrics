import torch
import os

save_dir = '/Users/joanacorreia/Desktop/AECD/Robustness-Metrics/models/Attacked_tensors/'

def evaluate_attack_with_logits(logits, labels):
    """
    Evaluate model accuracy using logits.
    """
    # Get predictions from logits (argmax across the class dimension)
    _, predicted = logits.max(1)

    # Compare predictions with ground truth labels
    correct = predicted.eq(labels).sum().item()

    # Calculate accuracy
    accuracy = 100.0 * correct / labels.size(0)
    return accuracy

def all_attacks(model, test_loader, attacks_to_test, model_name, save_dir=save_dir):
    for config in attacks_to_test:
        attack_name = config["name"]
        params = config["params"]
        attack_fn = config["model"]
        generate_and_save_attack_logits_with_labels(model, test_loader, attack_fn, attack_name, params, model_name, save_dir)

def generate_and_save_attack_logits_with_labels(model, test_loader, attack_fn, attack_name, params, model_name, save_dir=save_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_logits = []
    all_labels = []

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{attack_name}_logits_labels.pth")

    if not os.path.exists(save_path):
        print(f"Generating adversarial examples and saving logits for {model_name} using {attack_name}...")

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Generate adversarial images using torchattacks
            attack = attack_fn(model, **params)
            adv_images = attack(images, labels)  # Get adversarial images

            # Get logits for adversarial examples
            with torch.no_grad():
                logits = model(adv_images)  # Forward pass on adversarial images
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        # Concatenate all logits and labels
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Save the logits and labels
        torch.save({"logits": all_logits, "labels": all_labels}, save_path)

        print(f"Saved logits and labels to {save_path}")
        # Calculate accuracy or other metrics with logits
        adversarial_accuracy = evaluate_attack_with_logits(all_logits, all_labels)  # Pass logits and labels
        print(f"Accuracy on the test set after {attack_name} attack for {model_name}: {adversarial_accuracy:.2f}%")
    else:
        print(f"Attack {attack_name} for model {model_name} already generated. Skipping calculations")
