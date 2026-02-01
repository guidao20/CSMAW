import os
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from utils.dataset_utils import get_dataset
from utils.model_utils import get_models

@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        out = model(data)
        pred = out.argmax(dim=1)
        correct += (pred == target).sum().item()
        total += target.numel()
        
    return correct / total

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Training Script")
    parser.add_argument('--num_clients', type=int, required=True, help="Number of participating clients")
    parser.add_argument('--batch_size', type=int, default=64, help="Local batch size")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for the model")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of global communication rounds")
    parser.add_argument('--local_epoch', type=int, default=1, help="Number of local epochs per round")
    parser.add_argument('--data_dir', type=str, default='./data', help="Directory for dataset storage")
    parser.add_argument('--pretrained_dir', type=str, default='./pretrained_dir', help="Directory containing pre-trained weights")
    parser.add_argument('--pretrained', action='store_true', help="If set, load pre-trained weights. Otherwise train from scratch.")
    parser.add_argument('--artifacts_dir', type=str, required=True, help="Output directory for saved models")
    parser.add_argument('--seed', type=int, default=2026, help="Random seed for reproducibility")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset")
    parser.add_argument('--model', type=str, required=True, help="Model architecture name")

    args = parser.parse_args()
    
    print(f"\n====== Configuration for {os.path.basename(__file__)} ======")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("="*50)   

    artifacts_dir = args.artifacts_dir
    os.makedirs(artifacts_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed_all(args.seed)

    # load data
    train_dataset, test_dataset, _ = get_dataset(args.dataset, args.data_dir)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)

    # simulate IID data distribution by shuffling and splitting equally
    dataset_indices = np.arange(len(train_dataset))
    np.random.shuffle(dataset_indices)
    split = len(train_dataset) // args.num_clients
    
    client_indices = [dataset_indices[i*split:(i+1)*split] for i in range(args.num_clients)]
    client_loaders = [
        DataLoader(Subset(train_dataset, idxs), batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        for idxs in client_indices
    ]

    # model initialization
    ModelClass = get_models(args.model)
    global_tail = ModelClass(dataset_name=args.dataset).to(device)

    # 根据参数决定是否加载权重
    if args.pretrained:
        model_file = os.path.join(args.pretrained_dir, f'{args.dataset}_{args.model}_best.pth')
        print(f"Loading pre-trained weights from {model_file}...")
        if os.path.exists(model_file):
            pretrained_dict = torch.load(model_file, map_location=device)
            global_tail.load_state_dict(pretrained_dict)
            print("Pre-trained weights loaded successfully.")
        else:
            print(f"Warning: Pre-trained file {model_file} not found! Training from scratch.")
    else:
        print("No pre-trained weights requested. Training from scratch.")
    
    # initialize client models (local-side placeholders)
    client_tails = [ModelClass(dataset_name=args.dataset).to(device) for _ in range(args.num_clients)]
    final_client_tails = [ModelClass(dataset_name=args.dataset).to(device) for _ in range(args.num_clients)]

    print("=== Local training & FedAvg ===")
    # federated learning loop
    for num_round in range(args.num_rounds):
        tail_state_dicts = []
        print(f"\n[Federated round {num_round+1}]")
        
        # client local training phase
        for cid in range(args.num_clients):
            # synchronize client model with global weights
            client_tails[cid].load_state_dict(global_tail.state_dict())
            client_tails[cid].train()
            
            # initialize optimizer
            optimizer = optim.SGD(client_tails[cid].parameters(), lr=args.learning_rate, momentum=0.99)
            
            # local epochs
            for _ in range(args.local_epoch):
                for data, target in client_loaders[cid]:
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad(set_to_none=True) # set_to_none is slightly faster
                    out = client_tails[cid](data)
                    loss = nn.CrossEntropyLoss()(out, target)
                    loss.backward()
                    optimizer.step()
            
            # collect updated local weights
            tail_state_dicts.append({k: v.detach().cpu() for k, v in client_tails[cid].state_dict().items()})

        # server aggregation phase (FedAvg)
        avg_state_dict = {}
        for key in tail_state_dicts[0]:
            # simple averaging of weights
            avg_state_dict[key] = sum(sd[key] for sd in tail_state_dicts) / args.num_clients
        
        # update global model
        global_tail.load_state_dict(avg_state_dict)

        # evaluation
        acc = test_accuracy(global_tail, test_loader, device)
        print(f"Global TailNet acc after aggregation: {acc:.2%}")

        # store the specific state of clients at the very last round before aggregation/exit
        if num_round == args.num_rounds - 1:
            for cid in range(args.num_clients):
                final_client_tails[cid].load_state_dict(client_tails[cid].state_dict())

    torch.save({'state_dict': global_tail.cpu().state_dict()}, os.path.join(artifacts_dir, 'global_tail.pt'))
    
    for cid in range(args.num_clients):
        torch.save({'state_dict': final_client_tails[cid].cpu().state_dict()},
                   os.path.join(artifacts_dir, f'client_{cid}_tail.pt'))

    print("\nSaved:")
    print(f"  {os.path.join(artifacts_dir, 'global_tail.pt')}")
    print(f"  {args.num_clients} client model files: client_0_tail.pt ... client_{args.num_clients-1}_tail.pt")

if __name__ == "__main__":
    main()