import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn

from utils.dataset_utils import get_dataset
from utils.model_utils import get_models

def main():
    parser = argparse.ArgumentParser(description="Step 3: Traceability Assessment & Source Identification")
    parser.add_argument('--num_clients', type=int, required=True, help="Number of participating clients")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size for evaluation")
    parser.add_argument('--artifacts_dir', type=str, required=True, help="Directory containing generated artifacts from Step 1 & 2")
    parser.add_argument('--pgd_eps', type=float, default=0.8, help="Maximum perturbation budget (epsilon) used in PGD")
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
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)

    # load data
    print("Loading artifacts...")
    masks_ckpt = torch.load(os.path.join(artifacts_dir, 'masks.pt'), map_location='cpu')
    masks = masks_ckpt['all_masks']
    
    clean_ckpt = torch.load(os.path.join(artifacts_dir, 'clean_set.pt'), map_location='cpu')
    clean_images = clean_ckpt['clean_images']
    
    seqs_ckpt = torch.load(os.path.join(artifacts_dir, 'target_seqs.pt'), map_location='cpu')
    target_seqs = seqs_ckpt['target_seqs']   
    
    adv_pkg = torch.load(os.path.join(artifacts_dir, 'watermarks.pt'), map_location='cpu')
    adv_list = adv_pkg['adv_examples']
    
    # validate integrity of Step 2 output
    if 'adv_indices' not in adv_pkg:
        raise RuntimeError("Artifacts missing 'adv_indices'. Please re-run Step 2 with the updated code.")
    adv_indices = adv_pkg['adv_indices']

    num_clients = args.num_clients
    
    # model initialization
    ModelClass = get_models(args.model)
    ckpt = torch.load(os.path.join(artifacts_dir, 'global_tail.pt'), map_location='cpu')
    
    # load global model for traceability inference
    global_tail = ModelClass(dataset_name=args.dataset).to(device)
    global_tail.load_state_dict(ckpt['state_dict'])
    global_tail.eval()         

    # matrix to store targeted success rates (row: detector, col: source)
    tar_succ = [[0.0 for _ in range(num_clients)] for __ in range(num_clients)]
    
    print("\n=== Tracing (Global Tail) ===")
    
    with torch.no_grad():
        for cid in range(num_clients):
            print(f"\n[Head Client {cid+1} (Detector)]")
            
            # prepare target labels and mask for the current detector client
            head_target_seq = target_seqs[cid].to(device) 
            head_mask = masks[cid:cid+1, ...].to(device)
            
            # iterate through all adversarial sets
            for aid in range(num_clients):
                adv_data = adv_list[aid].to(device)
                
                # retrieve original dataset indices
                cur_indices = adv_indices[aid] 
                
                num_adv = adv_data.size(0)
                if num_adv == 0:
                    print(f"  Vs Adv Set {aid+1}: Empty (0 samples)")
                    continue

                total, success = 0, 0
                for i in range(0, num_adv, args.batch_size):
                    inputs = adv_data[i:i+args.batch_size]
                    batch_idxs = cur_indices[i:i+args.batch_size]

                    # apply the detector's mask to the incoming adversarial examples
                    inputs_masked = inputs * (1 - head_mask)
                    
                    # retrieve specific target labels for this batch
                    labels = head_target_seq[batch_idxs.to(device)] 
                    
                    preds = global_tail(inputs_masked)
                    pred_labels = preds.argmax(dim=1)
                    
                    total += labels.numel()
                    success += (pred_labels == labels).sum().item()
                
                succ_rate = success / total if total > 0 else 0
                tar_succ[cid][aid] = succ_rate
                # print(f"  -> Vs Adv Set {aid+1}: Success Rate = {succ_rate:.4f}")

    print("\n" + "="*40)
    print("=== Self-Identification Results (Diagonal) ===")
    
    self_detection_rates = [tar_succ[i][i] for i in range(num_clients)]
    

    for i, val in enumerate(self_detection_rates):
        print(f"Client {i+1} detected own samples: {val:.4%}")
    
    
if __name__ == "__main__":
    main()