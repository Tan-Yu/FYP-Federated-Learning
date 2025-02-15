import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.models import resnet18
import numpy as np
from collections import OrderedDict
import copy
import random
from typing import Dict, List, Optional, Tuple
import os
import json
from datetime import datetime
import csv
import pandas as pd
import shutil

def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')

device = get_device()

class BinaryImageDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        # data is expected to be a tensor of shape [N, H, W, C]
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Get image and convert from [H, W, C] to [C, H, W]
        image = self.data[idx].permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]
        label = self.targets[idx]

        if self.transform:
            # Remove ToPILImage and ToTensor from transform since we're already handling the format
            image = self.transform(image)

        return image, label

def create_binary_cifar10(train=True, classes=(3, 5)):  # 3: cat, 5: dog
    # Load CIFAR-10
    dataset = datasets.CIFAR10(
        root='./data',
        train=train,
        download=True,
        transform=None  # No transform here, we'll handle it in the Dataset class
    )

    # Get indices of the two classes we want
    idx = torch.tensor(dataset.targets) == classes[0]
    idx |= torch.tensor(dataset.targets) == classes[1]
    idx = torch.where(idx)[0]

    # Convert numpy arrays to tensors and maintain proper dimensions
    data = torch.tensor(np.stack([dataset.data[i] for i in idx.numpy()]))  # Shape: [N, H, W, C]
    targets = torch.tensor([dataset.targets[i] for i in idx])

    # Convert targets to binary (0 and 1)
    targets = (targets == classes[1]).long()

    return data, targets

# Define the image transformations
def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:
        return transforms.Compose([
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

def create_datasets(train_fraction=0.1):
    """Create train and validation datasets"""
    # Load binary CIFAR-10 (cats vs dogs)
    train_data, train_targets = create_binary_cifar10(train=True)
    test_data, test_targets = create_binary_cifar10(train=False)

    # Create datasets with appropriate transforms
    train_dataset = BinaryImageDataset(
        train_data, 
        train_targets,
        transform=get_transforms(train=True)
    )
    
    val_dataset = BinaryImageDataset(
        test_data,
        test_targets,
        transform=get_transforms(train=False)
    )

    # Create a smaller training set if requested
    if train_fraction < 1.0:
        num_train = int(len(train_dataset) * train_fraction)
        indices = torch.randperm(len(train_dataset))[:num_train]
        train_dataset = Subset(train_dataset, indices)

    return train_dataset, val_dataset


class VisualPromptEncoder(nn.Module):
    """
    Modified prompt encoder for visual tasks.
    Generates learnable spatial prompts that can be concatenated with image features.
    """
    def __init__(self, num_prompts, channels):
        super().__init__()
        self.num_prompts = num_prompts
        self.channels = channels
        
        # Initialize learnable prompt parameters (already 1x1 spatial size)
        self.prompt_embeddings = nn.Parameter(
            torch.randn(num_prompts * channels, 1, 1)
        )
        
        # Initialize with small random values
        nn.init.uniform_(self.prompt_embeddings, -0.02, 0.02)
        
    def forward(self, batch_size):
        # Expand prompts for batch size
        # [batch_size, num_prompts * channels, 1, 1]
        return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

class ModifiedResNet(nn.Module):
    """
    Modified ResNet that accepts additional prompt features.
    """
    def __init__(self, original_resnet):
        super().__init__()
        # Extract all layers before the final average pooling
        self.conv1 = original_resnet.conv1
        self.bn1 = original_resnet.bn1
        self.relu = original_resnet.relu
        self.maxpool = original_resnet.maxpool
        self.layer1 = original_resnet.layer1
        self.layer2 = original_resnet.layer2
        self.layer3 = original_resnet.layer3
        self.layer4 = original_resnet.layer4
        self.avgpool = original_resnet.avgpool
        
        # Remove the original fully connected layer
        del original_resnet.fc
        
    def forward(self, x, prompt_features=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Apply average pooling to get 1x1 spatial dimensions
        x = self.avgpool(x)  # Now x is [batch_size, channels, 1, 1]
        
        if prompt_features is not None:
            # Both x and prompt_features are now [batch_size, channels, 1, 1]
            x = torch.cat([x, prompt_features], dim=1)
        
        x = torch.flatten(x, 1)
        return x

class PrefixTuningModel(nn.Module):
    """
    Prefix tuning model adapted for image classification.
    """
    def __init__(self, num_prompts=4, num_classes=2):
        super().__init__()
        # Load pretrained ResNet-18
        original_resnet = resnet18(pretrained=True)
        
        # Initialize modified ResNet
        self.base_model = ModifiedResNet(original_resnet)
        
        # Set channel dimensions
        resnet_channels = 512  # ResNet-18's final channel dimension
        prompt_channels = 512  # Match ResNet's channel dimension
        
        # Initialize prompt encoder
        self.prompt_encoder = VisualPromptEncoder(
            num_prompts=num_prompts,
            channels=prompt_channels // num_prompts  # Divide channels among prompts
        )
        
        # Classifier input dimension is sum of ResNet features and prompt features
        total_channels = resnet_channels + prompt_channels
        self.classifier = nn.Linear(total_channels, num_classes)
        
        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
            
        # Make prompt encoder and classifier parameters trainable
        for param in self.prompt_encoder.parameters():
            param.requires_grad = True
        for param in self.classifier.parameters():
            param.requires_grad = True
    
    def get_prompt_params(self):
        """Returns only the trainable parameters (prompt encoder and classifier)"""
        return {
            name: param.clone().detach()
            for name, param in self.named_parameters()
            if 'prompt_encoder' in name or 'classifier' in name
        }
            
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Generate prompt features
        prompt_features = self.prompt_encoder(batch_size)
        
        # Get features from base model with prompts
        features = self.base_model(x, prompt_features)
        
        # Classification
        logits = self.classifier(features)
        
        return logits



class BaseClient:
    def __init__(self, model, train_dataset, val_dataset, device, client_id, **kwargs):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.device = device
        self.client_id = client_id
        self.criterion = nn.CrossEntropyLoss()

    def get_prompt_params(self):
        return {
            k: v.clone().detach()
            for k, v in self.model.named_parameters()
            if 'prompt_encoder' in k or 'classifier' in k
        }

    def evaluate(self):
        self.model.eval()
        val_loader = DataLoader(self.val_dataset, batch_size=32)
        correct = 0
        total = 0
        val_loss = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()
        
        accuracy = 100 * correct / total
        avg_loss = val_loss / len(val_loader)
        return accuracy, avg_loss

class FedAvgClient(BaseClient):
    def __init__(self, model, train_dataset, val_dataset, device, client_id, **kwargs):
        super().__init__(model, train_dataset, val_dataset, device, client_id, **kwargs)
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters() 
             if 'prompt_encoder' in n or 'classifier' in n],
            lr=1e-4  # Lower learning rate for image classification
        )

    def train(self, global_params=None, aggregation_type="fedavg", epochs=1, batch_size=32):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        accuracy, val_loss = self.evaluate()
        return total_loss / len(train_loader), accuracy, val_loss

class FedProxClient(BaseClient):
    def __init__(self, model, train_dataset, val_dataset, device, client_id, mu=0.01, **kwargs):
        super().__init__(model, train_dataset, val_dataset, device, client_id, **kwargs)
        self.mu = mu
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters() 
             if 'prompt_encoder' in n or 'classifier' in n],
            lr=1e-4
        )

    def train(self, global_params=None, aggregation_type="fedprox", epochs=1, batch_size=32):
        train_loader = DataLoader(
            self.train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                if global_params is not None:
                    proximal_term = 0
                    for name, param in self.model.get_prompt_params().items():
                        proximal_term += (self.mu / 2) * torch.norm(
                            param - global_params[name].to(self.device)
                        ) ** 2
                    loss += proximal_term
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        accuracy, val_loss = self.evaluate()
        return total_loss / len(train_loader), accuracy, val_loss
    
class MOONClient(BaseClient):
    def __init__(self, model, train_dataset, val_dataset, device, client_id, temperature=0.5, **kwargs):
        super().__init__(model, train_dataset, val_dataset, device, client_id, **kwargs)
        self.previous_model = None
        self.temperature = temperature
        
        # Projection head for contrastive learning
        self.projection = nn.Sequential(
            nn.Linear(512, 256),  # Adjusted for ResNet-18 feature size
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)
        
        # Initialize optimizer with projection head parameters
        self.optimizer = torch.optim.AdamW(
            list(self.projection.parameters()) + 
            [p for n, p in self.model.named_parameters() 
             if 'prompt_encoder' in n or 'classifier' in n],
            lr=1e-4
        )

    def train(self, global_params=None, aggregation_type="moon", epochs=1, batch_size=32):
        if self.previous_model is None:
            self.previous_model = copy.deepcopy(self.model)
            
        global_model = copy.deepcopy(self.model)
        if global_params is not None:
            global_model.load_state_dict(global_params)
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.model.train()
        total_loss = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Get features from different models
                current_features = self.model.base_model(images)
                with torch.no_grad():
                    prev_features = self.previous_model.base_model(images)
                    global_features = global_model.base_model(images)
                
                # Task loss
                outputs = self.model(images)
                task_loss = self.criterion(outputs, labels)
                
                # Contrastive loss
                current_rep = self.projection(current_features)
                prev_rep = self.projection(prev_features)
                global_rep = self.projection(global_features)
                
                current_rep = F.normalize(current_rep, dim=1)
                prev_rep = F.normalize(prev_rep, dim=1)
                global_rep = F.normalize(global_rep, dim=1)
                
                pos_score = torch.sum(current_rep * prev_rep, dim=1)
                neg_score = torch.sum(current_rep * global_rep, dim=1)
                
                logits = torch.cat([pos_score.unsqueeze(1), neg_score.unsqueeze(1)], dim=1)
                logits /= self.temperature
                contra_labels = torch.zeros(logits.shape[0], dtype=torch.long, device=self.device)
                
                contra_loss = F.cross_entropy(logits, contra_labels)
                loss = task_loss + 0.1 * contra_loss
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        self.previous_model = copy.deepcopy(self.model)
        
        accuracy, val_loss = self.evaluate()
        return total_loss / len(train_loader), accuracy, val_loss

class FedDynClient(BaseClient):
    def __init__(self, model, train_dataset, val_dataset, device, client_id, alpha=0.01, **kwargs):
        super().__init__(model, train_dataset, val_dataset, device, client_id, **kwargs)
        self.alpha = alpha
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters() 
             if 'prompt_encoder' in n or 'classifier' in n],
            lr=1e-4
        )

    def train(self, global_params=None, aggregation_type="feddyn", epochs=1, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        total_loss = 0
        param_gradients = {
            name: torch.zeros_like(param.data, device=self.device)
            for name, param in self.model.named_parameters()
            if 'prompt_encoder' in name or 'classifier' in name
        }
        
        num_batches = 0
        for epoch in range(epochs):
            for images, labels in train_loader:
                num_batches += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                if global_params is not None:
                    proximal_term = 0
                    for name, param in self.model.named_parameters():
                        if 'prompt_encoder' in name or 'classifier' in name:
                            proximal_term += (self.alpha / 2) * torch.norm(
                                param - global_params[name].to(self.device)
                            ) ** 2
                    loss += proximal_term
                
                loss.backward()
                
                for name, param in self.model.named_parameters():
                    if ('prompt_encoder' in name or 'classifier' in name) and param.grad is not None:
                        param_gradients[name] += param.grad.data
                
                self.optimizer.step()
                total_loss += loss.item()
        
        param_gradients = {k: v / num_batches for k, v in param_gradients.items()}
        
        accuracy, val_loss = self.evaluate()
        return total_loss / len(train_loader), accuracy, val_loss, param_gradients

class SCAFFOLDClient(BaseClient):
    def __init__(self, model, train_dataset, val_dataset, device, client_id, **kwargs):
        super().__init__(model, train_dataset, val_dataset, device, client_id, **kwargs)
        self.control_variate = None
        self.server_control_variate = None
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters() 
             if 'prompt_encoder' in n or 'classifier' in n],
            lr=1e-4
        )
        
    def initialize_control_variate(self):
        self.control_variate = {
            k: torch.zeros_like(v).to(self.device)
            for k, v in self.model.state_dict().items()
            if 'prompt_encoder' in k or 'classifier' in k
        }
    
    def train(self, global_params=None, server_control_variate=None, 
              aggregation_type="scaffold", epochs=1, batch_size=32):
        if self.control_variate is None:
            self.initialize_control_variate()
        
        if server_control_variate is not None:
            self.server_control_variate = {
                k: v.to(self.device) for k, v in server_control_variate.items()
            }
        
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        
        initial_params = {
            k: v.clone() for k, v in self.model.state_dict().items()
            if 'prompt_encoder' in k or 'classifier' in k
        }
        
        total_loss = 0
        num_steps = 0
        
        self.model.train()
        for epoch in range(epochs):
            for images, labels in train_loader:
                num_steps += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                
                if self.server_control_variate is not None:
                    for name, param in self.model.named_parameters():
                        if ('prompt_encoder' in name or 'classifier' in name) and param.grad is not None:
                            correction = (self.server_control_variate[name] - 
                                       self.control_variate[name])
                            param.grad = param.grad - correction
                
                self.optimizer.step()
                total_loss += loss.item()
        
        control_delta = {}
        final_params = self.model.state_dict()
        for key in self.control_variate.keys():
            param_diff = (final_params[key] - initial_params[key]) / num_steps
            control_delta[key] = (
                self.server_control_variate[key] - 
                self.control_variate[key] + 
                param_diff
            )
            self.control_variate[key] += control_delta[key]
        
        accuracy, val_loss = self.evaluate()
        return total_loss / len(train_loader), accuracy, val_loss, control_delta
    

class BaseServer:
    def __init__(self, initial_model):
        self.global_model = copy.deepcopy(initial_model)
        self.current_round = 0
        self.device = get_device()

    def get_model_params(self):
        return copy.deepcopy(self.global_model.state_dict())

class FedAvgServer(BaseServer):
    def __init__(self, initial_model, **kwargs):
        super().__init__(initial_model)
        
    def aggregate_models(self, client_models, aggregation_type="fedavg"):
        self.current_round += 1
        aggregated_dict = self.global_model.state_dict()
        
        for key in aggregated_dict.keys():
            if 'prompt_encoder' in key or 'classifier' in key:
                aggregated_dict[key] = torch.stack(
                    [client_models[i][key].float() for i in range(len(client_models))], 
                    0
                ).mean(0)
        
        self.global_model.load_state_dict(aggregated_dict)
        return copy.deepcopy(self.global_model.state_dict())

class FedProxServer(BaseServer):
    def __init__(self, initial_model, mu=0.01, **kwargs):
        super().__init__(initial_model)
        self.mu = mu
        
    def aggregate_models(self, client_models, aggregation_type="fedprox"):
        self.current_round += 1
        aggregated_dict = self.global_model.state_dict()
        
        for key in aggregated_dict.keys():
            if 'prompt_encoder' in key or 'classifier' in key:
                aggregated_dict[key] = torch.stack(
                    [client_models[i][key].float() for i in range(len(client_models))], 
                    0
                ).mean(0)
        
        self.global_model.load_state_dict(aggregated_dict)
        return copy.deepcopy(self.global_model.state_dict())

class FedOptServer(BaseServer):
    def __init__(self, initial_model, beta1=0.9, beta2=0.999, tau=1e-3, **kwargs):
        super().__init__(initial_model)
        self.beta1 = beta1
        self.beta2 = beta2
        self.tau = tau
        self.m = None
        self.v = None
        
    def aggregate_models(self, client_models, aggregation_type="fedopt"):
        self.current_round += 1
        aggregated_dict = self.global_model.state_dict()
        
        if self.m is None:
            self.m = {k: torch.zeros_like(v) for k, v in aggregated_dict.items() 
                     if 'prompt_encoder' in k or 'classifier' in k}
            self.v = {k: torch.zeros_like(v) for k, v in aggregated_dict.items() 
                     if 'prompt_encoder' in k or 'classifier' in k}
            
        for key in aggregated_dict.keys():
            if 'prompt_encoder' in key or 'classifier' in key:
                delta = torch.stack([client_models[i][key].float() - aggregated_dict[key].float() 
                                   for i in range(len(client_models))], 0).mean(0)
                
                self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * delta
                self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * delta.pow(2)
                
                aggregated_dict[key] += self.tau * self.m[key] / (self.v[key].sqrt() + 1e-8)
        
        self.global_model.load_state_dict(aggregated_dict)
        return copy.deepcopy(self.global_model.state_dict())

class FedDynServer(BaseServer):
    def __init__(self, initial_model, alpha=0.01, **kwargs):
        super().__init__(initial_model)
        self.alpha = alpha
        self.global_grads = None
        
    def aggregate_models(self, client_models, client_grads, aggregation_type="feddyn"):
        self.current_round += 1
        aggregated_dict = self.global_model.state_dict()
        
        if self.global_grads is None:
            self.global_grads = {
                k: torch.zeros_like(v) 
                for k, v in aggregated_dict.items() 
                if 'prompt_encoder' in k or 'classifier' in k
            }
        
        for key in aggregated_dict.keys():
            if 'prompt_encoder' in key or 'classifier' in key:
                if all(key in grad for grad in client_grads):
                    grad_sum = torch.stack([grad[key].to(self.device) for grad in client_grads], 0).sum(0)
                    self.global_grads[key] += self.alpha * grad_sum
                    
                    aggregated_dict[key] = (torch.stack([client_models[i][key].float() 
                                           for i in range(len(client_models))], 0).mean(0) - 
                                           self.global_grads[key])
        
        self.global_model.load_state_dict(aggregated_dict)
        return copy.deepcopy(self.global_model.state_dict())

class SCAFFOLDServer(BaseServer):
    def __init__(self, initial_model, **kwargs):
        super().__init__(initial_model)
        self.global_control_variate = None
        self.device = get_device()
        
    def initialize_control_variate(self):
        self.global_control_variate = {
            k: torch.zeros_like(v).to(self.device)
            for k, v in self.global_model.state_dict().items()
            if 'prompt_encoder' in k or 'classifier' in k
        }
        
    def aggregate_models(self, client_models, client_control_deltas, aggregation_type="scaffold"):
        self.current_round += 1
        aggregated_dict = self.global_model.state_dict()
        
        if self.global_control_variate is None:
            self.initialize_control_variate()
        
        for key in aggregated_dict.keys():
            if 'prompt_encoder' in key or 'classifier' in key:
                aggregated_dict[key] = torch.stack(
                    [client_models[i][key].float() for i in range(len(client_models))], 
                    0
                ).mean(0)
                
                delta_sum = torch.stack(
                    [delta[key].float() for delta in client_control_deltas], 
                    0
                ).mean(0)
                self.global_control_variate[key] += delta_sum
        
        self.global_model.load_state_dict(aggregated_dict)
        return (copy.deepcopy(self.global_model.state_dict()), 
                copy.deepcopy(self.global_control_variate))
        
# Add these classes right after SCAFFOLDServer and before FederatedLogger

class FedNovaServer(BaseServer):
    def __init__(self, initial_model, **kwargs):
        super().__init__(initial_model)
        
    def aggregate_models(self, client_models, local_steps, aggregation_type="fednova"):
        self.current_round += 1
        aggregated_dict = self.global_model.state_dict()
        
        # Calculate total steps across all clients
        total_steps = sum(local_steps)
        normalized_weights = [steps / total_steps for steps in local_steps]
        
        # Aggregate parameters with normalized weights
        for key in aggregated_dict.keys():
            if 'prompt_encoder' in key or 'classifier' in key:
                weighted_sum = torch.zeros_like(aggregated_dict[key])
                for client_idx, weight in enumerate(normalized_weights):
                    weighted_sum += weight * client_models[client_idx][key].float()
                aggregated_dict[key] = weighted_sum
        
        self.global_model.load_state_dict(aggregated_dict)
        return copy.deepcopy(self.global_model.state_dict())

class FedNovaClient(BaseClient):
    def __init__(self, model, train_dataset, val_dataset, device, client_id, **kwargs):
        super().__init__(model, train_dataset, val_dataset, device, client_id, **kwargs)
        self.optimizer = torch.optim.AdamW(
            [p for n, p in self.model.named_parameters() 
             if 'prompt_encoder' in n or 'classifier' in n],
            lr=1e-4
        )

    def train(self, global_params=None, aggregation_type="fednova", epochs=1, batch_size=32):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        total_steps = len(train_loader) * epochs
        
        self.model.train()
        total_loss = 0
        actual_steps = 0
        
        for epoch in range(epochs):
            for images, labels in train_loader:
                actual_steps += 1
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
        
        accuracy, val_loss = self.evaluate()
        return total_loss / len(train_loader), accuracy, val_loss, actual_steps
    
class FederatedLogger:
    def __init__(self, base_dir: str = "logs"):
        self.base_dir = base_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directory structure
        self.experiment_dir = os.path.join(base_dir, self.timestamp)
        self.client_dir = os.path.join(self.experiment_dir, "client_logs")
        self.global_dir = os.path.join(self.experiment_dir, "global_logs")
        
        os.makedirs(self.client_dir, exist_ok=True)
        os.makedirs(self.global_dir, exist_ok=True)
        
        # Initialize file paths
        self.global_csv = os.path.join(self.global_dir, "global_metrics.csv")
        self.summary_csv = os.path.join(self.experiment_dir, "experiment_summary.csv")
        
        # Initialize CSV files
        self._initialize_csv_files()
    
    def _initialize_csv_files(self):
        # Global metrics CSV
        with open(self.global_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'algorithm', 'round', 'global_accuracy',
                'avg_client_accuracy', 'avg_client_loss'
            ])
        
        # Summary CSV
        with open(self.summary_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'algorithm', 'best_accuracy', 'final_accuracy',
                'avg_training_loss', 'total_rounds', 'num_clients'
            ])
    
    def create_client_log(self, algorithm: str) -> str:
        client_csv = os.path.join(self.client_dir, f"{algorithm}_client_metrics.csv")
        with open(client_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'round', 'client_id', 'training_loss',
                'validation_loss', 'accuracy'
            ])
        return client_csv
    
    def log_client_metrics(self, file_path: str, round_num: int, client_id: int,
                          train_loss: float, val_loss: float, accuracy: float):
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                round_num,
                client_id,
                train_loss,
                val_loss,
                accuracy
            ])
    
    def log_global_metrics(self, algorithm: str, round_num: int, global_accuracy: float,
                          avg_client_accuracy: float, avg_client_loss: float):
        with open(self.global_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                algorithm,
                round_num,
                global_accuracy,
                avg_client_accuracy,
                avg_client_loss
            ])
    
    def log_experiment_summary(self, algorithm: str, best_accuracy: float,
                             final_accuracy: float, avg_training_loss: float,
                             total_rounds: int, num_clients: int):
        with open(self.summary_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                algorithm,
                best_accuracy,
                final_accuracy,
                avg_training_loss,
                total_rounds,
                num_clients
            ])
    
    def save_configuration(self, config: Dict):
        config_path = os.path.join(self.experiment_dir, "configuration.csv")
        with open(config_path, 'w', newline='') as f:
            writer = csv.writer(f)
            for key, value in config.items():
                writer.writerow([key, value])

class CheckpointManager:
    def __init__(self, base_dir: str = "checkpoints", peft_method: str = "lora"):
        self.base_dir = base_dir
        self.peft_method = peft_method.lower()
        
        # Create PEFT-specific directory
        self.peft_dir = os.path.join(base_dir, self.peft_method)
        os.makedirs(self.peft_dir, exist_ok=True)
        
        # Find most recent checkpoint directory
        checkpoint_dirs = [
            os.path.join(self.peft_dir, d) for d in os.listdir(self.peft_dir) 
            if os.path.isdir(os.path.join(self.peft_dir, d))
        ]
        
        if checkpoint_dirs:
            self.checkpoint_dir = max(checkpoint_dirs, key=os.path.getmtime)
            print(f"Found existing checkpoint directory: {self.checkpoint_dir}")
        else:
            self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.checkpoint_dir = os.path.join(self.peft_dir, self.timestamp)
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            print(f"Created new checkpoint directory: {self.checkpoint_dir}")
        
        self.metadata_file = os.path.join(self.checkpoint_dir, "metadata.json")
        self.checkpoint_counter = self._get_next_checkpoint_number()

    def _get_next_checkpoint_number(self):
        if not os.path.exists(self.metadata_file):
            return 1
            
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        checkpoints = metadata.get("checkpoints", [])
        if not checkpoints:
            return 1
            
        max_num = max(
            int(checkpoint["checkpoint_num"]) 
            for checkpoint in checkpoints
        )
        return max_num + 1

    def _generate_checkpoint_name(self, checkpoint_num: int, round_num: int, algorithm: str, 
                                model_params: dict = None) -> str:
        components = [
            f"{checkpoint_num:02d}",
            f"r{round_num:03d}",
            self.peft_method,
            algorithm.lower(),
        ]
        
        if model_params:
            for key, value in model_params.items():
                if isinstance(value, float):
                    param_str = f"{key}{value:.3f}".replace('.', 'p')
                else:
                    param_str = f"{key}{value}".replace('.', 'p')
                components.append(param_str)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        components.append(timestamp)
        
        return f"{'_'.join(components)}.pt"

    def _cleanup_previous_checkpoint(self, current_algorithm: str):
        if not os.path.exists(self.metadata_file):
            return
            
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
            
        checkpoints = metadata.get("checkpoints", [])
        algo_checkpoints = [cp for cp in checkpoints if cp["algorithm"] == current_algorithm]
        
        if len(algo_checkpoints) <= 1:
            return
            
        algo_checkpoints.sort(key=lambda x: int(x["checkpoint_num"]))
        checkpoints_to_delete = algo_checkpoints[:-1]
        
        updated_checkpoints = []
        for checkpoint in checkpoints:
            if checkpoint in checkpoints_to_delete:
                checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint["filename"])
                try:
                    os.remove(checkpoint_path)
                    print(f"Deleted previous checkpoint: {checkpoint['filename']}")
                except FileNotFoundError:
                    print(f"Warning: Checkpoint file not found: {checkpoint['filename']}")
            else:
                updated_checkpoints.append(checkpoint)
        
        metadata["checkpoints"] = updated_checkpoints
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def save_checkpoint(self, round_num: int, algorithm: str, global_model_state: dict,
                       clients_states: List[dict], server_state: dict = None,
                       extra_state: dict = None):
        checkpoint_name = self._generate_checkpoint_name(
            self.checkpoint_counter, round_num, algorithm
        )
        checkpoint_path = os.path.join(self.checkpoint_dir, checkpoint_name)
        
        checkpoint = {
            'metadata': {
                'checkpoint_num': self.checkpoint_counter,
                'round': round_num,
                'algorithm': algorithm,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            },
            'global_model_state': global_model_state,
            'clients_states': clients_states,
            'server_state': server_state,
            'extra_state': extra_state
        }
        
        torch.save(checkpoint, checkpoint_path)
        
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {'checkpoints': []}
        
        metadata['checkpoints'].append({
            'filename': checkpoint_name,
            'checkpoint_num': self.checkpoint_counter,
            'round': round_num,
            'algorithm': algorithm,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
        
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self._cleanup_previous_checkpoint(algorithm)
        self.checkpoint_counter += 1

    def load_latest_checkpoint(self):
        if not os.path.exists(self.metadata_file):
            return None, None
        
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        
        if not metadata['checkpoints']:
            return None, None
        
        checkpoints = metadata['checkpoints']
        checkpoints.sort(key=lambda x: x['checkpoint_num'])
        latest_checkpoint = checkpoints[-1]
        
        checkpoint_path = os.path.join(self.checkpoint_dir, latest_checkpoint['filename'])
        
        if not os.path.exists(checkpoint_path):
            return None, None
        
        checkpoint = torch.load(checkpoint_path)
        return checkpoint, latest_checkpoint['algorithm']


def create_algorithm_state_dict(algorithm: str, server, clients_list: list) -> dict:
    state_dict = {}
    
    if algorithm == "fedopt":
        state_dict["m"] = server.m
        state_dict["v"] = server.v
    elif algorithm == "scaffold":
        state_dict["global_control_variate"] = server.global_control_variate
        state_dict["client_control_variates"] = [
            client.control_variate for client in clients_list
        ]
    elif algorithm == "moon":
        state_dict["previous_models"] = [
            client.previous_model.state_dict() if client.previous_model else None
            for client in clients_list
        ]
    elif algorithm == "feddyn":
        state_dict["global_grads"] = server.global_grads
    
    return state_dict

def load_algorithm_state(algorithm: str, state_dict: dict, server, clients_list: list):
    if not state_dict:
        return
    
    if algorithm == "fedopt":
        server.m = state_dict["m"]
        server.v = state_dict["v"]
    elif algorithm == "scaffold":
        server.global_control_variate = state_dict["global_control_variate"]
        for client, control_variate in zip(clients_list, state_dict["client_control_variates"]):
            client.control_variate = control_variate
    elif algorithm == "moon":
        for client, prev_model_state in zip(clients_list, state_dict["previous_models"]):
            if prev_model_state:
                client.previous_model.load_state_dict(prev_model_state)
    elif algorithm == "feddyn":
        server.global_grads = state_dict["global_grads"]

    

def main():
    # Set random seeds
    set_seeds(42)
    
    # Initialize logging and checkpointing
    logger = FederatedLogger(base_dir="federated_logs")
    checkpoint_manager = CheckpointManager(base_dir="federated_checkpoints", peft_method="prefix_catdog")
    
    # Configure parameters
    NUM_CLIENTS = 5
    num_rounds = 10
    batch_size = 16
    local_epochs = 2
    num_prompts = 4
    
    # Save configuration
    config = {
        "num_clients": NUM_CLIENTS,
        "num_rounds": num_rounds,
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "num_prompts": num_prompts,
        "device": str(device)
    }
    logger.save_configuration(config)
    
    print("Loading and preparing datasets...")
    train_dataset, val_dataset = create_datasets(train_fraction=0.1)
    
    # Split training data for clients
    total_size = len(train_dataset)
    size_per_client = total_size // NUM_CLIENTS
    client_train_datasets = [
        Subset(train_dataset, 
               range(i * size_per_client, (i + 1) * size_per_client))
        for i in range(NUM_CLIENTS)
    ]
    
    # Initialize model
    print("Initializing model...")
    model = PrefixTuningModel(
        num_prompts=num_prompts,
        num_classes=2
    ).to(device)
    
    # Define available algorithms
    servers = {
        "fedavg": FedAvgServer,
        "fedprox": FedProxServer,
        "fedopt": FedOptServer,
        "feddyn": FedDynServer,
        "scaffold": SCAFFOLDServer,
        "moon": FedAvgServer,
        "fednova": FedNovaServer
    }
    
    clients = {
        "fedavg": FedAvgClient,
        "fedprox": FedProxClient,
        "moon": MOONClient,
        "scaffold": SCAFFOLDClient,
        "fedopt": FedAvgClient,
        "feddyn": FedDynClient,
        "fednova": FedNovaClient
    }
    
    # Algorithm-specific parameters
    algorithm_params = {
        "fedprox": {"client_params": {"mu": 0.01}, "server_params": {}},
        "fedopt": {"client_params": {}, "server_params": {"beta1": 0.9, "beta2": 0.999, "tau": 1e-3}},
        "moon": {"client_params": {"temperature": 0.5}, "server_params": {}},
        "feddyn": {"client_params": {"alpha": 0.01}, "server_params": {"alpha": 0.01}},
        "scaffold": {"client_params": {}, "server_params": {}},
        "fedavg": {"client_params": {}, "server_params": {}},
        "fednova": {"client_params": {}, "server_params": {}}
    }
    
    # Define order of algorithms to run
    aggregation_types = [
        "fedavg", "fedprox", "fedopt", "moon", "feddyn", "scaffold", "fednova"
    ]
    
    # Load latest checkpoint if available
    checkpoint_data, last_algorithm = checkpoint_manager.load_latest_checkpoint()
    start_round = 0
    completed_algorithms = set()
    
    if checkpoint_data is not None:
        print(f"Found checkpoint from round {checkpoint_data['metadata']['round']}")
        metadata_file = os.path.join(checkpoint_manager.checkpoint_dir, "metadata.json")
        
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            checkpoints = metadata.get("checkpoints", [])
            algorithm_rounds = {}
            
            for checkpoint in checkpoints:
                alg = checkpoint["algorithm"]
                round_num = checkpoint["round"]
                algorithm_rounds[alg] = round_num
                
                if round_num == num_rounds:
                    completed_algorithms.add(alg)
            
            if last_algorithm in algorithm_rounds and last_algorithm not in completed_algorithms:
                start_round = algorithm_rounds[last_algorithm]
                if start_round >= num_rounds:
                    start_round = 0
                print(f"Continuing {last_algorithm} from round {start_round}")
    
        aggregation_types = [alg for alg in aggregation_types if alg not in completed_algorithms]
        if last_algorithm in aggregation_types:
            idx = aggregation_types.index(last_algorithm)
            aggregation_types = aggregation_types[idx:]
    
    if not aggregation_types:
        print("All algorithms have completed training!")
        return
    
    print(f"Will run the following algorithms: {aggregation_types}")
    
    results = {}
    client_logs = {
        agg_type: logger.create_client_log(agg_type)
        for agg_type in aggregation_types
    }
    
    # Run experiments for each algorithm
    for agg_type in aggregation_types:
        print(f"\nStarting training with {agg_type.upper()}")
        print("-" * 50)
        
        params = algorithm_params.get(agg_type, {
            "client_params": {},
            "server_params": {}
        })
        
        server_class = servers.get(agg_type)
        server = server_class(model, **params["server_params"])
        
        client_class = clients.get(agg_type)
        clients_list = [
            client_class(
                copy.deepcopy(model),
                train_dataset,
                val_dataset,
                device,
                i,
                **params["client_params"]
            )
            for i, train_dataset in enumerate(client_train_datasets)
        ]
        
        best_accuracy = 0
        round_metrics = []
        
        # Load checkpoint state if available
        if checkpoint_data is not None and agg_type == last_algorithm:
            server.global_model.load_state_dict(checkpoint_data['global_model_state'])
            for client, state in zip(clients_list, checkpoint_data['clients_states']):
                client.model.load_state_dict(state)
            if checkpoint_data['server_state']:
                load_algorithm_state(agg_type, checkpoint_data['server_state'], 
                                  server, clients_list)
            if checkpoint_data['extra_state']:
                best_accuracy = checkpoint_data['extra_state'].get('best_accuracy', 0)
                round_metrics = checkpoint_data['extra_state'].get('round_metrics', [])
        
        # Initialize algorithm-specific components
        if agg_type == "scaffold":
            server.initialize_control_variate()
            for client in clients_list:
                client.initialize_control_variate()
        
        # Training loop
        for round in range(start_round, num_rounds):
            print(f"\nRound {round + 1}/{num_rounds}")
            print("-" * 30)
            
            # Get current global parameters
            if agg_type == "scaffold":
                global_params = (server.global_model.state_dict(), server.global_control_variate)
            else:
                global_params = server.global_model.state_dict()
            
            # Client training
            client_models = []
            client_metrics = []
            round_accuracies = []
            total_train_loss = 0
            
            for client in clients_list:
                if agg_type == "scaffold":
                    train_loss, accuracy, val_loss, control_delta = client.train(
                        *global_params,
                        aggregation_type=agg_type,
                        epochs=local_epochs,
                        batch_size=batch_size
                    )
                    client_metrics.append(control_delta)
                elif agg_type == "feddyn":
                    train_loss, accuracy, val_loss, gradients = client.train(
                        global_params,
                        aggregation_type=agg_type,
                        epochs=local_epochs,
                        batch_size=batch_size
                    )
                    client_metrics.append(gradients)
                elif agg_type == "fednova":  # Add this condition
                    train_loss, accuracy, val_loss, steps = client.train(
                        global_params,
                        aggregation_type=agg_type,
                        epochs=local_epochs,
                        batch_size=batch_size
                    )
                    client_metrics.append(steps)
                else:
                    train_loss, accuracy, val_loss = client.train(
                        global_params,
                        aggregation_type=agg_type,
                        epochs=local_epochs,
                        batch_size=batch_size
                    )

                
                print(f"Client {client.client_id}:")
                print(f"  Training Loss: {train_loss:.4f}")
                print(f"  Validation Loss: {val_loss:.4f}")
                print(f"  Validation Accuracy: {accuracy:.2f}%")
                
                client_models.append(client.get_prompt_params())
                round_accuracies.append(accuracy)
                total_train_loss += train_loss
            
            # Calculate round metrics
            avg_accuracy = sum(round_accuracies) / len(round_accuracies)
            avg_train_loss = total_train_loss / len(clients_list)
            
            # Log global metrics
            logger.log_global_metrics(
                agg_type,
                round + 1,
                avg_accuracy,
                avg_accuracy,
                avg_train_loss
            )
            
            round_metrics.append({
                'round': round + 1,
                'avg_accuracy': avg_accuracy,
                'client_accuracies': round_accuracies
            })
            
            print(f"\nRound {round + 1} Average Accuracy: {avg_accuracy:.2f}%")
            
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                print(f"New best accuracy!")
            
            # Server aggregation
            if agg_type == "scaffold":
                global_params, server_control_variate = server.aggregate_models(
                    client_models,
                    client_metrics
                )
            elif agg_type in ["feddyn", "fednova"]:  # Update this condition
                global_params = server.aggregate_models(client_models, client_metrics)
            else:
                global_params = server.aggregate_models(client_models)
            
            # Save checkpoint
            checkpoint_manager.save_checkpoint(
                round_num=round + 1,
                algorithm=agg_type,
                global_model_state=server.global_model.state_dict(),
                clients_states=[client.model.state_dict() for client in clients_list],
                server_state=create_algorithm_state_dict(agg_type, server, clients_list),
                extra_state={
                    'best_accuracy': best_accuracy,
                    'round_metrics': round_metrics
                }
            )
        
        # Log experiment summary
        if round_metrics:
            logger.log_experiment_summary(
                agg_type,
                best_accuracy,
                round_metrics[-1]['avg_accuracy'],
                sum([m['avg_accuracy'] for m in round_metrics]) / len(round_metrics),
                num_rounds,
                NUM_CLIENTS
            )
            
            results[agg_type] = {
                'best_accuracy': best_accuracy,
                'round_metrics': round_metrics
            }
        else:
            print(f"Warning: No metrics available for {agg_type}")
        
        print(f"\n{agg_type.upper()} Final Best Accuracy: {best_accuracy:.2f}%")
        start_round = 0
    
    # Print comparative results
    print("\nComparative Results:")
    print("-" * 50)
    for alg, res in results.items():
        print(f"{alg.upper()}:")
        print(f"  Best Accuracy: {res['best_accuracy']:.2f}%")
        if res['round_metrics']:
            print(f"  Final Round Avg Accuracy: {res['round_metrics'][-1]['avg_accuracy']:.2f}%")
        print("-" * 30)

if __name__ == "__main__":
    main()