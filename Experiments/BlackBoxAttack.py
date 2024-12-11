import torch
from torch.utils.data import DataLoader, TensorDataset
"""Define RayS attack which is a query-based black-box attack method."""

def RaySAttack(model, images, labels, epsilon, query_budget, device):
    model.eval()  # Ensure the model is in evaluation mode

    batch_size = images.size(0)
    perturbations = torch.zeros_like(images, device=device)

    # Initial random directions
    directions = torch.randn_like(images).sign().to(device)

    for query_count in range(query_budget):
        # Positive and negative perturbations
        pos_images = torch.clamp(images + perturbations + epsilon * directions, 0, 1)
        neg_images = torch.clamp(images + perturbations - epsilon * directions, 0, 1)

        # Query the model
        pos_outputs = model(pos_images)
        neg_outputs = model(neg_images)

        # Get predictions
        _, pos_preds = pos_outputs.max(1)
        _, neg_preds = neg_outputs.max(1)

        # Update perturbations based on misclassification
        pos_misclassified = pos_preds != labels
        neg_misclassified = neg_preds != labels

        perturbations[pos_misclassified] += epsilon * directions[pos_misclassified]
        perturbations[neg_misclassified] -= epsilon * directions[neg_misclassified]

        # Stop if all images are misclassified
        if pos_misclassified.sum() + neg_misclassified.sum() == batch_size:
            break

    # Apply the final perturbations
    adversarial_examples = torch.clamp(images + perturbations, 0, 1)
    return adversarial_examples

def generate_rays_adv_loader(model, data_loader, epsilon, query_budget, device):
    model.eval()

    adv_images_list = []
    labels_list = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)

        batch_adv_images = []
        for i in range(images.size(0)):
            # Run RaySAttack on a single image
            adv_image = RaySAttack(
                model=model,
                images=images[i:i+1],  # Single image
                labels=labels[i:i+1],
                epsilon=epsilon,
                query_budget=query_budget,
                device=device
            )
            batch_adv_images.append(adv_image)

        # Concatenate adversarial examples for the batch
        adv_images_list.append(torch.cat(batch_adv_images, dim=0).cpu())
        labels_list.append(labels.cpu())

    # Concatenate all adversarial examples and labels
    adv_images_all = torch.cat(adv_images_list, dim=0)
    labels_all = torch.cat(labels_list, dim=0)

    # Create a new DataLoader for adversarial examples
    adv_dataset = TensorDataset(adv_images_all, labels_all)
    adv_loader = DataLoader(adv_dataset, batch_size=data_loader.batch_size, shuffle=False)

    return adv_loader