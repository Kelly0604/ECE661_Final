# FGSM and MIM functions sourced from 2021 ICCV paper, "On the Robustness of Vision Transformers to Adversarial Examples." 
# Original code can be found in: https://github.com/MetaMain/ViTRobust/tree/main
# Remaining functions including PGD, APGD, BPDA, and CW are adopted based on these two functions

#Attack wrappers class for FGSM and MIM (no extra library implementation) to be used in conjunction with 
#the adaptive black-box attack 
import torch 
import DataManagerPytorch as DMP
import torchvision
from torch.utils.data import DataLoader, TensorDataset
#from torchattacks import APGD
import torch.optim as optim

#Native (no attack library) implementation of the FGSM attack in Pytorch 
def FGSMNativePytorch(device, dataLoader, model, epsilonMax, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    #Go through each sample 
    tracker = 0
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        #print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xDataTemp = torch.from_numpy(xData.cpu().detach().numpy()).to(device)
        yData = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        xDataTemp.requires_grad = True
        # Forward pass the data through the model
        output = model(xDataTemp)
        # Calculate the loss
        loss = torch.nn.CrossEntropyLoss()
        # Zero all existing gradients
        model.zero_grad()
        # Calculate gradients of model in backward pass
        cost = loss(output, yData).to(device)
        cost.backward()
        # Collect datagrad
        #xDataGrad = xDataTemp.grad.data
        ###Here we actual compute the adversarial sample 
        # Collect the element-wise sign of the data gradient
        signDataGrad = xDataTemp.grad.data.sign()
        # Create the perturbed image by adjusting each pixel of the input image
        #print("xData:", xData.is_cuda)
        #print("SignGrad:", signDataGrad.is_cuda)
        if targeted == True:
            perturbedImage = xData - epsilonMax*signDataGrad.cpu().detach() #Go negative of gradient
        else:
            perturbedImage = xData + epsilonMax*signDataGrad.cpu().detach()
        # Adding clipping to maintain the range
        perturbedImage = torch.clamp(perturbedImage, clipMin, clipMax)
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = perturbedImage[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index
        #Not sure if we need this but do some memory clean up 
        del xDataTemp
        del signDataGrad
        torch.cuda.empty_cache()
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None) #use the same batch size as the original loader
    return advLoader

#Native (no attack library) implementation of the MIM attack in Pytorch 
#This is only for the L-infinty norm and cross entropy loss function 
def MIMNativePytorch(device, dataLoader, model, decayFactor, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval() #Change model to evaluation mode for the attack 
    #Generate variables for storing the adversarial examples 
    numSamples = len(dataLoader.dataset) #Get the total number of samples to attack
    xShape = DMP.GetOutputShape(dataLoader) #Get the shape of the input (there may be easier way to do this)
    xAdv = torch.zeros(numSamples, xShape[0], xShape[1], xShape[2])
    yClean = torch.zeros(numSamples)
    advSampleIndex = 0 
    batchSize = 0 #just do dummy initalization, will be filled in later
    loss = torch.nn.CrossEntropyLoss()
    tracker = 0
    #Go through each sample 
    for xData, yData in dataLoader:
        batchSize = xData.shape[0] #Get the batch size so we know indexing for saving later
        tracker = tracker + batchSize
        print("Processing up to sample=", tracker)
        #Put the data from the batch onto the device 
        xAdvCurrent = xData.to(device)
        yCurrent = yData.type(torch.LongTensor).to(device)
        # Set requires_grad attribute of tensor. Important for attack. (Pytorch comment, not mine) 
        #Initalize memory for the gradient momentum
        gMomentum = torch.zeros(batchSize, xShape[0], xShape[1], xShape[2])
        #Do the attack for a number of steps
        for attackStep in range(0, numSteps):   
            xAdvCurrent.requires_grad = True
            outputs = model(xAdvCurrent)
            model.zero_grad()
            cost = loss(outputs, yCurrent).to(device)
            cost.backward()
            #Update momentum term 
            gMomentum = decayFactor*gMomentum + GradientNormalizedByL1(xAdvCurrent.grad)
            #Update the adversarial sample 
            if targeted == True:
                advTemp = xAdvCurrent - (epsilonStep*torch.sign(gMomentum)).to(device)
            else:
                advTemp = xAdvCurrent + (epsilonStep*torch.sign(gMomentum)).to(device)
            #Adding clipping to maintain the range
            xAdvCurrent = torch.clamp(advTemp, min=clipMin, max=clipMax).detach_()
        #Save the adversarial images from the batch 
        for j in range(0, batchSize):
            xAdv[advSampleIndex] = xAdvCurrent[j]
            yClean[advSampleIndex] = yData[j]
            advSampleIndex = advSampleIndex+1 #increment the sample index 
    #All samples processed, now time to save in a dataloader and return 
    advLoader = DMP.TensorToDataLoader(xAdv, yClean, transforms= None, batchSize= dataLoader.batch_size, randomizer=None)
    return advLoader

def PGDNativePytorch(device, dataLoader, model, epsilonMax, epsilonStep, numSteps, clipMin, clipMax, targeted):
    model.eval()
    numSamples = len(dataLoader.dataset)
    xShape = next(iter(dataLoader))[0].shape[1:]
    xAdv = torch.zeros(numSamples, *xShape)
    yClean = torch.zeros(numSamples, dtype=torch.long)
    
    advSampleIndex = 0
    for xData, yData in dataLoader:
        xData, yData = xData.to(device), yData.to(device)
        batchSize = xData.shape[0]
        advImages = xData.clone().detach()

        # PGD iterations
        for _ in range(numSteps):
            advImages.requires_grad = True
            outputs = model(advImages)
            loss = torch.nn.CrossEntropyLoss()(outputs, yData)
            model.zero_grad()
            loss.backward()

            # Generate adversarial perturbations
            gradSign = advImages.grad.sign()
            if targeted:
                advImages = advImages - epsilonStep * gradSign  # Targeted: decrease towards target
            else:
                advImages = advImages + epsilonStep * gradSign  # Untargeted: increase loss

            # Project adversarial examples to epsilon-ball and clip to valid range
            advImages = torch.clamp(advImages, xData - epsilonMax, xData + epsilonMax)
            advImages = torch.clamp(advImages, clipMin, clipMax).detach()

        # Save adversarial examples
        for j in range(batchSize):
            xAdv[advSampleIndex] = advImages[j].cpu()
            yClean[advSampleIndex] = yData[j].cpu()
            advSampleIndex += 1

    # Return DataLoader with adversarial examples
    advLoader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xAdv, yClean), 
        batch_size=dataLoader.batch_size,
        shuffle=False
    )
    return advLoader

def CWPytorch(device, dataLoader, model, c=1.0, kappa=0, numSteps=1000, lr=0.01, clipMin=0.0, clipMax=1.0):
    model.eval()  # Set the model to evaluation mode

    # Initialize variables to store adversarial examples
    numSamples = len(dataLoader.dataset)
    xShape = next(iter(dataLoader))[0].shape[1:]  # Get input shape from DataLoader
    xAdv = torch.zeros(numSamples, *xShape)
    yClean = torch.zeros(numSamples, dtype=torch.long)
    advSampleIndex = 0

    for xData, yData in dataLoader:
        xData, yData = xData.to(device), yData.to(device)
        batchSize = xData.size(0)
        
        # Initialize adversarial perturbation
        delta = torch.zeros_like(xData, requires_grad=True).to(device)

        # Define optimizer for delta
        optimizer = optim.Adam([delta], lr=lr)

        for _ in range(numSteps):
            advImages = torch.clamp(xData + delta, clipMin, clipMax)  # Add perturbation and clip to valid range

            # Forward pass
            outputs = model(advImages)

            # Define C&W loss function
            real = outputs.gather(1, yData.view(-1, 1)).squeeze()  # Correct class scores
            one_hot = torch.eye(outputs.size(1), device=device)[yData]
            other = torch.max((outputs - one_hot * 1e4), dim=1)[0]  # Max of incorrect class scores
            f = torch.clamp(real - other + kappa, min=0)  # Confidence margin

            l2_norm = torch.sum(delta ** 2)
            loss = l2_norm + c * torch.sum(f)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save adversarial examples
        for j in range(batchSize):
            xAdv[advSampleIndex] = torch.clamp(xData[j] + delta[j], clipMin, clipMax).cpu()
            yClean[advSampleIndex] = yData[j].cpu()
            advSampleIndex += 1

    # Return a DataLoader with adversarial examples
    advLoader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(xAdv, yClean),
        batch_size=dataLoader.batch_size,
        shuffle=False
    )
    return advLoader

def GradientNormalizedByL1(gradient):
    #Do some basic error checking first
    if gradient.shape[1] != 3:
        raise ValueError("Shape of gradient is not consistent with an RGB image.")
    #basic variable setup
    batchSize = gradient.shape[0]
    colorChannelNum = gradient.shape[1]
    imgRows = gradient.shape[2]
    imgCols = gradient.shape[3]
    gradientNormalized = torch.zeros(batchSize, colorChannelNum, imgRows, imgCols)
    #Compute the L1 gradient for each color channel and normalize by the gradient 
    #Go through each color channel and compute the L1 norm
    for i in range(0, batchSize):
        for c in range(0, colorChannelNum):
           norm = torch.linalg.norm(gradient[i,c], ord=1)
           gradientNormalized[i,c] = gradient[i,c]/norm #divide the color channel by the norm
    return gradientNormalized
