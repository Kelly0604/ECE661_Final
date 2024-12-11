# Generate attacks using predefined attack functions and preprocessed CIFAR-10 data 
# and evaluate the model on the adversarial examples
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from WhiteBoxAttacks import FGSMNativePytorch, MIMNativePytorch, PGDNativePytorch, APGDNativePytorch, BPDANativePytorch, CWPytorch

from ResNetPytorch import ResNet56

def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet56()
    model_path = './saved_model/resnet56_cifar10.pth'
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model, device

def prepare_data():
    # Data preprocessing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Data augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # Normalize to [-1, 1]
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    # DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    return train_loader, test_loader

def evaluate(model, data_loader):
    correct = 0
    total = 0
    device = next(model.parameters()).device
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')

def main():
    model, device = load_model()
    _, test_loader = prepare_data()

    # Example using FGSM attack method 
    epsilon = 0.031  # Perturbation magnitude
    clip_min = 0.0
    clip_max = 1.0
    targeted = False

    # Generate adversarial examples using FGSMNativePytorch
    adv_loader_FGSM = FGSMNativePytorch(
        device=device,
        dataLoader=test_loader,
        model=model,
        epsilonMax=epsilon,
        clipMin=clip_min,
        clipMax=clip_max,
        targeted=targeted
    )

    # Evaluate the model on adversarial examples
    print("Evaluating on C&W adversarial examples:")
    evaluate(model, adv_loader_FGSM)

    # Example using C&W attack method
    c = 1.0  # Regularization constant
    kappa = 0  # Confidence margin
    num_steps = 100
    lr = 0.01
    clip_min = 0.0
    clip_max = 1.0

    # Generate C&W adversarial examples
    adv_loader_CW = CWPytorch(
        device=device,
        dataLoader=test_loader,
        model=model,
        c=c,
        kappa=kappa,
        numSteps=num_steps,
        lr=lr,
        clipMin=clip_min,
        clipMax=clip_max
    )
    # Evaluate the model on adversarial examples
    print("Evaluating on C&W adversarial examples:")
    evaluate(model, adv_loader_CW)

if __name__ == '__main__':
    main()
