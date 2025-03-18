import argparse
from datetime import datetime
import os
import torch
import torch.utils.data
import torch.optim as optim
import torchvision
import torch.distributed as dist

from base.datasets import CocoScenes
from base.models import get_model_instance_segmentation
from base.training import train_coco_scenes
from base.result_manager import ResultManager

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    # dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

    
def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)

# collate_fn needs for batch
def collate_fn(batch):
    batch = list(filter(lambda x: x[1] is not None, batch))
    return tuple(zip(*batch))


def run_ddp(rank, args, model, my_dataset, optimizer, n_epochs, verbose, result_manager):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, args.world_size)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	my_dataset,
    	num_replicas=args.world_size,
    	rank=rank
    )

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                            batch_size=int(args.batch_size),
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=train_sampler,
                                            collate_fn=collate_fn)

    train_coco_scenes(model=model, trainloader=data_loader, args=args, optimizer=optimizer, device=rank, 
           n_epochs=n_epochs, verbose=verbose, result_manager=result_manager)

    cleanup()

def run_gpu(args, model, my_dataset, optimizer, n_epochs, verbose, result_manager, device, num_workers):
    train_sampler = torch.utils.data.distributed.DistributedSampler(
    	my_dataset,
    	num_replicas=1,
    	rank=0
    )

    # own DataLoader
    data_loader = torch.utils.data.DataLoader(my_dataset,
                                            batch_size=int(args.batch_size),
                                            num_workers=num_workers,
                                            pin_memory=True,
                                            sampler=train_sampler,
                                            collate_fn=collate_fn)

    train_coco_scenes(model=model, trainloader=data_loader, args=args, optimizer=optimizer, device=device, 
           n_epochs=n_epochs, verbose=verbose, result_manager=result_manager)

if __name__ == '__main__':
    home_path = os.path.expanduser('~')

    c_time = datetime.now().strftime("%d-%m-%y-%H%M%S")

    # Instantiate the parser
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--output_dir', help='Path to output directory.', default=f'{home_path}/projects/coco_results/{c_time}')
    parser.add_argument(
        '--n_epochs', help='Number of epochs for training.', default=30)
    parser.add_argument(
        '--n_processes', help='Number of processes to use.', default=18)
    parser.add_argument(
        '--nodes', help='Number of nodes.', default=1)
    parser.add_argument(
        '--batch_size', help='Batch size for training.', default=1)
    parser.add_argument(
        '--optimizer', help='Optimizer to use for training', default='adam')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')

    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes


    torch.manual_seed(42)

    # path to your own data and coco file
    train_data_dir = 'data/COCO/processed/train2017'
    train_coco = 'data/COCO/processed/annotations/instances_train2017.json'

    # create own Dataset
    my_dataset = CocoScenes(root=train_data_dir,
                            annotation=train_coco,
                            transforms=get_transform()
                            )



    num_classes = 100
    model = get_model_instance_segmentation(num_classes)

    # parameters
    params = [p for p in model.parameters() if p.requires_grad]
    
    # Optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(params, lr=0.001, momentum=0.9)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(params, lr=0.001)
    elif args.optimizer == 'rmsprop':
        optimizer = optim.RMSprop(params, lr=0.001, momentum=0.9)

    result_manager = ResultManager(root=args.output_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    run_gpu(args, model, my_dataset, optimizer, int(args.n_epochs), True, result_manager, device=device, num_workers=int(args.n_processes))