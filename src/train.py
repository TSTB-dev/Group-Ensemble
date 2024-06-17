import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import wandb
import logging as log
log.basicConfig(level=log.INFO)
import argparse

from dataset import load_cifar10_dataloader, load_cifar100_dataloader, load_mnist_dataloader
from models import get_model
from lr import CyclicCosineLRScheduler

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--model", type=str, default="resnet18")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--num_ensembles", type=int, default=4)
    parser.add_argument("--scheduler_type", type=str, default="cyclic_cosine")
    parser.add_argument("--max_lr", type=float, default=0.1)
    parser.add_argument("--min_lr", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--optimizer", type=str, default="sgd")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--proj_name", type=str, default="group_ensemble")
    parser.add_argument("--save_dir", type=str, default="models")

    args = parser.parse_args()
    return args

def train(args):
    log.info(f"Starting training with args: {args}")
    
    # fix seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device(args.device)
    
    wandb.init(project=args.proj_name)
    
    log.info(f"Loading dataset: {args.dataset}")
    if args.dataset == "cifar10":
        train_loader, test_loader = load_cifar10_dataloader(args)
    elif args.dataset == "cifar100":
        train_loader, test_loader = load_cifar100_dataloader(args)
    elif args.dataset == "mnist":
        train_loader, test_loader = load_mnist_dataloader(args)
    else:
        raise ValueError("Invalid dataset")
    num_iters_per_epoch = len(train_loader)
    num_iters_total = num_iters_per_epoch * args.num_epochs
    log.info(f"Number of iterations per epoch: {num_iters_per_epoch}")
    log.info(f"Total number of iterations: {num_iters_total}")
    log.info(f"Dataset loaded")
    
    log.info(f"Creating model: {args.model}")
    model = get_model(args)
    model = model.to(device)
    log.info(f"Model created")
    
    log.info(f"Creating optimizer: {args.optimizer}")
    if args.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise ValueError("Invalid optimizer")
    
    log.info(f"Creating scheduler: {args.scheduler_type}")
    if args.scheduler_type == "cyclic_cosine":
        scheduler = CyclicCosineLRScheduler(optimizer, args.max_lr, args.min_lr, num_iters_total)
    else:
        raise ValueError("Invalid scheduler")
    
    log.info("Starting training")
    
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(args.num_epochs):
        for i, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)  # (B, G, num_classes)
            loss = loss_fn(outputs.view(-1, outputs.size(-1)), targets.repeat_interleave(outputs.size(1))).mean()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if i % args.log_interval == 0:
                log.info(f"Epoch: {epoch}/{args.num_epochs}, Iter: {i}/{num_iters_per_epoch}, Loss: {loss.item()}")
                wandb.log({"loss": loss.item()})
    log.info("Training finished")
    
    save_path = os.path.join(f"{wandb.run.dir}", f"{args.dataset}_{args.model}.pth")
    torch.save(model.state_dict(), save_path)
    log.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    args = get_args()
    train(args)
    

