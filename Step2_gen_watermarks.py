import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from utils.dataset_utils import get_dataset
from utils.model_utils import get_models

def _random_derangement(rng: np.random.RandomState, num_classes) -> np.ndarray:
    base = np.arange(num_classes, dtype=np.int64)
    while True:
        perm = rng.permutation(base)
        if not np.any(perm == base):
            return perm

def generate_target_sequences_from_true(true_labels: torch.Tensor, num_clients: int, num_classes: int, seed: int = 2025):
    assert true_labels.dim() == 1
    rng = np.random.RandomState(seed)
    used = set()
    derangements = []
    
    # Generate unique label mappings for each client
    for _ in range(num_clients):
        while True:
            der = _random_derangement(rng, num_classes)
            key = tuple(der.tolist())
            if key not in used:
                used.add(key)
                derangements.append(der)
                break
    
    true_np = true_labels.cpu().numpy().astype(np.int64)
    seqs = []
    for der in derangements:
        seq_np = der[true_np]
        seqs.append(torch.from_numpy(seq_np).long())
    return seqs

def generate_nonoverlapping_masks(num_clients, shape):
    flat_dim = int(np.prod(shape))
    indices = np.random.permutation(flat_dim)
    masks = torch.zeros((num_clients, flat_dim), dtype=torch.float32)
    size_per_client = flat_dim // num_clients
    
    for i in range(num_clients):
        start = i * size_per_client
        end = (i + 1) * size_per_client if i != num_clients - 1 else flat_dim
        masks[i, indices[start:end]] = 1.0
        
    masks = masks.view(num_clients, *shape)
    return masks

def pgd_attack(model, images, target_labels, eps, alpha, steps, mask, device):
    images = images.clone().detach().to(device)
    ori_images = images.clone().detach()
    target_labels = target_labels.to(device)
    mask = mask.to(device)
    images.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    
    for _ in range(steps):
        # Apply mask during forward pass
        out = model(images * (1 - mask))
        loss = criterion(out, target_labels)
        loss.backward()
        
        # Update images
        adv_images = images - alpha * images.grad.sign()
        eta = torch.clamp(adv_images - ori_images, min=-eps, max=eps)
        images = ori_images + eta
        images = torch.clamp(images, 0, 1)
        
        images = images * (1 - mask) + ori_images * mask
        
        images = images.detach_()
        images.requires_grad = True
        
    return images

def main():
    parser = argparse.ArgumentParser(description="Step 2: Generate Targeted Adversarial Examples")
    parser.add_argument('--num_clients', type=int, required=True, help="Number of participating clients")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for processing")
    parser.add_argument('--cleanset_max', type=int, default=300, help="Maximum size of the clean reference dataset")
    parser.add_argument('--pgd_eps', type=float, default=0.8, help="Maximum perturbation budget (epsilon)")
    parser.add_argument('--pgd_alpha', type=float, default=0.04, help="Step size (alpha) for PGD")
    parser.add_argument('--pgd_steps', type=int, default=200, help="Number of PGD optimization steps")
    parser.add_argument('--adv_target_count', type=int, default=200, help="Number of successful adversarial examples to save per client")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory for dataset storage")
    parser.add_argument('--artifacts_dir', type=str, required=True, help="Directory to save generated artifacts")
    parser.add_argument('--seed', type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--model', type=str, required=True, help="Model architecture name")
    args = parser.parse_args()

    artifacts_dir = args.artifacts_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)

    # load test data
    _, test_dataset, num_classes = get_dataset(args.dataset, args.data_dir)
    indices = np.arange(len(test_dataset))
    np.random.shuffle(indices)
    split = len(test_dataset) // args.num_clients
    client_indices = [indices[i*split:(i+1)*split] for i in range(args.num_clients)]
    client_loaders = [DataLoader(Subset(test_dataset, idxs), batch_size=args.batch_size, shuffle=False) for idxs in client_indices]    

    # build clean reference dataset
    print("Building clean reference dataset...")
    clean_images, true_labels = [], []
    for cid in range(args.num_clients):
        for data, label in client_loaders[cid]:
            clean_images.append(data)
            true_labels.append(label)
            if sum(x.size(0) for x in clean_images) >= args.cleanset_max: break
        if sum(x.size(0) for x in clean_images) >= args.cleanset_max: break
    
    clean_images = torch.cat(clean_images, dim=0)[:args.cleanset_max]
    true_labels  = torch.cat(true_labels,  dim=0)[:args.cleanset_max]
    print(f"Clean set built: N={clean_images.size(0)}")

    # generate targets & masks
    target_seqs = generate_target_sequences_from_true(true_labels, args.num_clients, num_classes, args.seed)
    mask_shape = tuple(clean_images[0].shape)
    all_masks = generate_nonoverlapping_masks(args.num_clients, mask_shape)

    # load client models
    ModelClass = get_models(args.model)
    client_models = []
    for cid in range(args.num_clients):
        ckpt = torch.load(os.path.join(artifacts_dir, f'client_{cid}_tail.pt'), map_location='cuda')
        m = ModelClass(dataset_name=args.dataset).to(device)
        m.load_state_dict(ckpt['state_dict'])
        m.eval()
        client_models.append(m)

    adv_examples = []
    adv_indices = []
    success_rates = []

    # main attack loop
    for cid in range(args.num_clients):
        print(f"\n[Client {cid+1}/{args.num_clients}] Processing...")
        y_target_seq = target_seqs[cid]
        mask_c = all_masks[cid:cid+1, ...]
        model_c = client_models[cid]
        active_mask = torch.zeros_like(mask_c) 
        
        # generate adversarial examples
        adv_batch = pgd_attack(model_c, clean_images, y_target_seq, args.pgd_eps, args.pgd_alpha, args.pgd_steps, active_mask, device)
        
        # filter: keep only successful attacks
        successful_indices_list = []
        y_target_dev = y_target_seq.to(device)
        mask_dev = active_mask.to(device)

        with torch.no_grad():
            for i in range(0, adv_batch.size(0), args.batch_size):
                x_batch = adv_batch[i:i+args.batch_size] 
                y_batch = y_target_dev[i:i+args.batch_size]
                x_batch_masked = x_batch * (1 - mask_dev) 
                
                preds = model_c(x_batch_masked).argmax(dim=1)
                matches = (preds == y_batch)
                
                if matches.sum() > 0:
                    batch_indices = torch.where(matches)[0] + i 
                    successful_indices_list.append(batch_indices)
        
        if successful_indices_list:
            successful_indices = torch.cat(successful_indices_list)
        else:
            successful_indices = torch.tensor([], dtype=torch.long, device=device)

        num_successful = successful_indices.size(0)
        succ_rate = num_successful / adv_batch.size(0)
        
        # select subset of successful examples
        if num_successful > args.adv_target_count:
            print(f"  Success: {num_successful}. Saving top {args.adv_target_count}.")
            final_indices = successful_indices[:args.adv_target_count]
        else:
            print(f"  Success: {num_successful}. Saving all.")
            final_indices = successful_indices
            
        adv_examples.append(adv_batch[final_indices].cpu())
        adv_indices.append(final_indices.cpu())
        success_rates.append(succ_rate)
        print(f"  Rate: {succ_rate:.2%}")

    torch.save({'clean_images': clean_images.cpu(), 'true_labels': true_labels.cpu()}, os.path.join(artifacts_dir, 'clean_set.pt'))
    torch.save({'all_masks': all_masks.cpu()}, os.path.join(artifacts_dir, 'masks.pt'))
    torch.save({'target_seqs': [t.cpu() for t in target_seqs]}, os.path.join(artifacts_dir, 'target_seqs.pt'))
    
    torch.save({
        'adv_examples': [ae.cpu() for ae in adv_examples],
        'adv_indices': [idx.cpu() for idx in adv_indices], 
        'success_rates': success_rates
    }, os.path.join(artifacts_dir, 'watermarks.pt'))

    print("\nDone. Saved artifacts with indices mapping.")

if __name__ == "__main__":
    main()