import os
import argparse
import datetime
import re
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP

import clip
from datasets import load_dataset
from timm.utils import AverageMeter

from utils import set_seed, Logger, set_logger, reduce_tensor
from utils import (
    UnlabeledDatasetV6,
    build_transforms,
    robust_PLCA,
    MomentumUpdate,
    MemoryBank,
    freeze_norm_layer,
)


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="ViT-B/32")

parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--split", type=str, default="test")

parser.add_argument("--iters", type=int, default=1)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--max_len", type=int, default=10000, help="Max length of memory bank")

parser.add_argument("--extend", type=int, default=8)
parser.add_argument("--repeat", type=int, default=1)
parser.add_argument("--pi", type=float, default=0.9)
parser.add_argument("--eval_interval", type=int, default=15)

parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--save_model", action="store_true", default=False)
parser.add_argument("--save_path", type=str, default="./results/name/")
parser.add_argument('--local_rank', type=int, default=-1, help='node rank for distributed training')
parser.add_argument("--local-rank", type=int, default=-1, help='node rank for distributed training')
args = parser.parse_args()


DEFAULT_PROMPT_TEMPLATE = "a photo of a {}"
REMOTE_SENSING_PROMPT_TEMPLATE = "a satellite image of {}"

DATASET_CONFIGS = {
    "arampacha/rsicd": {
        "aliases": ["rsicd"],
        "prompt_template": REMOTE_SENSING_PROMPT_TEMPLATE,
    },
    "blanchon/EuroSAT_RGB": {
        "aliases": ["eurosat", "eurosat_rgb"],
        "prompt_template": REMOTE_SENSING_PROMPT_TEMPLATE,
    },
    "timm/resisc45": {
        "aliases": ["resisc45", "resisc-45", "nwpu-resisc45"],
        "prompt_template": REMOTE_SENSING_PROMPT_TEMPLATE,
    },
    "blanchon/PatternNet": {
        "aliases": ["patternnet"],
        "prompt_template": REMOTE_SENSING_PROMPT_TEMPLATE,
    },
    "jonathan-roberts1/MLRSNet": {
        "aliases": ["mlrsnet"],
        "prompt_template": REMOTE_SENSING_PROMPT_TEMPLATE,
    },
}

DATASET_ALIASES = {
    alias: dataset_name
    for dataset_name, config in DATASET_CONFIGS.items()
    for alias in config["aliases"]
}

SPLIT_ALIASES = {
    "train": ["train"],
    "test": ["test"],
    "validation": ["validation", "valid", "val"],
    "valid": ["valid", "validation", "val"],
    "val": ["val", "validation", "valid"],
}


def resolve_dataset_name(dataset_name):
    return DATASET_ALIASES.get(dataset_name, dataset_name)


def get_dataset_config(dataset_name):
    return DATASET_CONFIGS.get(dataset_name, {})


def resolve_split_name(split_name, available_splits):
    if split_name in available_splits:
        return split_name

    for candidate in SPLIT_ALIASES.get(split_name, []):
        if candidate in available_splits:
            return candidate

    raise ValueError(f"Split '{split_name}' not found. Available splits: {sorted(available_splits)}")


def normalize_class_name(class_name):
    return class_name.replace("_", " ").replace("-", " ").strip()


def parse_rsicd_class_name(filename):
    filename = os.path.basename(filename)
    filename = os.path.splitext(filename)[0]
    filename = re.sub(r"_\d+$", "", filename)
    return normalize_class_name(filename)


def infer_dataset_spec(dataset_name, dataset):
    features = dataset.features

    if "image" in features:
        image_key = "image"
    elif "img" in features:
        image_key = "img"
    else:
        raise NotImplementedError("No image feature found in the dataset.")

    if dataset_name == "arampacha/rsicd":
        if "filename" not in features:
            raise NotImplementedError("RSICD requires a 'filename' field to derive scene labels.")

        class_names = sorted({parse_rsicd_class_name(filename) for filename in dataset["filename"]})
        return {
            "image_key": image_key,
            "label_mode": "filename_prefix",
            "class_names": class_names,
            "label_to_index": {label: idx for idx, label in enumerate(class_names)},
            "prompt_template": get_dataset_config(dataset_name).get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
        }

    if "label" in features:
        label_feature = features["label"]
        label_key = "label"
    elif "fine_label" in features:
        label_feature = features["fine_label"]
        label_key = "fine_label"
    else:
        raise NotImplementedError("No supported label feature found in the dataset.")

    if hasattr(label_feature, "names"):
        class_names = [normalize_class_name(name) for name in label_feature.names]
        label_mode = "single_label"
    elif hasattr(label_feature, "feature") and hasattr(label_feature.feature, "names"):
        class_names = [normalize_class_name(name) for name in label_feature.feature.names]
        label_mode = "multi_label"
    else:
        raise NotImplementedError("Unsupported label schema in the dataset.")

    return {
        "image_key": image_key,
        "label_key": label_key,
        "label_mode": label_mode,
        "class_names": class_names,
        "prompt_template": get_dataset_config(dataset_name).get("prompt_template", DEFAULT_PROMPT_TEMPLATE),
    }


def build_dataset_transform(preprocess, dataset_spec):
    def dataset_transform(examples):
        images = list(examples[dataset_spec["image_key"]])
        examples["img"] = images
        examples["image"] = [preprocess(image) for image in images]

        if dataset_spec["label_mode"] == "single_label":
            examples["label_targets"] = [[int(label)] for label in examples[dataset_spec["label_key"]]]
        elif dataset_spec["label_mode"] == "multi_label":
            examples["label_targets"] = [list(map(int, labels)) for labels in examples[dataset_spec["label_key"]]]
        elif dataset_spec["label_mode"] == "filename_prefix":
            examples["label_targets"] = [
                [dataset_spec["label_to_index"][parse_rsicd_class_name(filename)]]
                for filename in examples["filename"]
            ]
        else:
            raise NotImplementedError(f"Unsupported label mode: {dataset_spec['label_mode']}")

        return examples

    return dataset_transform


def load_hf_dataset(dataset_name, split_name):
    local_candidates = [
        os.path.join("./data/datasets", args.dataset),
        os.path.join("./data/datasets", dataset_name),
    ]

    dataset = None
    for source in local_candidates:
        try:
            dataset = load_dataset(source, split=split_name)
            break
        except Exception:
            continue

    if dataset is None:
        dataset = load_dataset(dataset_name, split=split_name)

    return dataset


def build_text_inputs(class_names, prompt_template):
    return torch.cat([clip.tokenize(prompt_template.format(class_name)) for class_name in class_names])


def compute_accuracy(predictions, label_targets):
    correct = 0
    for prediction, targets in zip(predictions.tolist(), label_targets):
        correct += int(prediction in targets)
    return predictions.new_tensor(correct / max(len(label_targets), 1), dtype=torch.float32)


# Initialize the environment
if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ['WORLD_SIZE'])
    print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
else:
    rank = -1
    world_size = -1

torch.cuda.set_device(args.local_rank)
dist.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank, timeout=datetime.timedelta(seconds=10800))
dist.barrier()

args.batch_size = args.batch_size // world_size
device = "cuda" if torch.cuda.is_available() else "cpu"
seed = args.seed + dist.get_rank()
set_seed(seed)

args.iter_batch = args.batch_size
if args.batch_size < 32:
    args.batch_size = 32
if args.iter_batch * args.extend < 32:
    args.extend = 32 // args.iter_batch


# Create the logger
experiment_name = f"{args.model}_{args.dataset}_{args.split}".replace("/", "-")
if dist.get_rank() == 0:
    logger = Logger(os.path.join(args.save_path, f"{experiment_name}.log"))
    logger.create_config(args)
    logger.info(args)
    set_logger(logger)


# Load the model
model, preprocess = clip.load(args.model, device=device, jit=False)
model.to(torch.float32)
model_without_ddp = model
model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)


# Create the datasets
adapt_transform = build_transforms(preprocess)
dataset_name = resolve_dataset_name(args.dataset)

try:
    split_dataset = load_hf_dataset(dataset_name, args.split)
    resolved_split = args.split
except Exception:
    available_splits = set(load_dataset(dataset_name).keys())
    resolved_split = resolve_split_name(args.split, available_splits)
    split_dataset = load_hf_dataset(dataset_name, resolved_split)

dataset_spec = infer_dataset_spec(dataset_name, split_dataset)
class_names = dataset_spec["class_names"]
text_inputs = build_text_inputs(class_names, dataset_spec["prompt_template"])
test_transform = build_dataset_transform(preprocess, dataset_spec)

train_dataset = split_dataset
test_dataset = load_hf_dataset(dataset_name, resolved_split)

train_dataset.set_transform(test_transform)
test_dataset.set_transform(test_transform)

if dist.get_rank() == 0:
    logger.info(f"Resolved dataset source: {dataset_name}, split: {resolved_split}")


# Create the dataloaders
def collate_fn(examples):
    batch = {}
    batch["image"] = torch.stack([example["image"] for example in examples])
    batch["img"] = [example["img"] for example in examples]
    batch["label_targets"] = [example["label_targets"] for example in examples]
    if all(len(targets) == 1 for targets in batch["label_targets"]):
        batch["label"] = torch.tensor([targets[0] for targets in batch["label_targets"]])
    return batch

train_sampler = torch.utils.data.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
train_loader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.iter_batch, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)
test_sampler = torch.utils.data.DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
test_loader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, collate_fn=collate_fn)


# Create optimizer, scheduler and criterion
param_group = [
    {"params": model_without_ddp.parameters(), "lr": args.lr}]
optimizer = torch.optim.SGD(param_group, weight_decay=1e-3)
momentum = MomentumUpdate(model_without_ddp)

def criterion(outputs, targets):
    loss_i = -targets * torch.log_softmax(outputs, dim=0)
    loss_t = -targets * torch.log_softmax(outputs, dim=1)
    loss = (loss_i + loss_t) / 2.
    loss = torch.sum(loss, dim=-1)
    loss = torch.mean(loss)
    return loss


# Compute the pseudo labels
memory_bank = MemoryBank(args.max_len)

def compute_pseudo_labels(dataset, preprocess, model, repeat=args.repeat, verbose=True, extend=args.extend):
    model.eval()

    def clip_predict(images, class_names):
        image_features = model.encode_image(images.to(device))
        text_features = model.encode_text(text_inputs.to(device))

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        soft_labels = (100.0 * image_features_norm @ text_features_norm.T).softmax(dim=-1)

        return soft_labels, image_features
    
    dataset.set_pseudo_labels(
        define_func=robust_PLCA,
        model_predict=clip_predict,
        preprocess=preprocess,
        extend=extend,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        store_loc=True,
        bank=memory_bank.get_all(),
        K=50,
        gamma=3,
        mode='l2',
        repeat=repeat,
        device=device,
        return_dst=True,
        verbose=verbose,
    )

    features = dataset.get_features()
    pseudo_labels = dataset.get_pseudo_labels()
    memory_bank.update(features, pseudo_labels)
    if verbose: print(f"Memory Bank Size: {memory_bank.size()}")
    dist.barrier()


# Test the model
max_acc_meter = AverageMeter()
model.eval()
with torch.no_grad():
    tbar = tqdm(test_loader, dynamic_ncols=True) if dist.get_rank() == 0 else test_loader
    for batch in tbar:
        images = batch["image"].to(device)
        label_targets = batch["label_targets"]

        image_features = model_without_ddp.encode_image(images)
        text_features = model_without_ddp.encode_text(text_inputs.to(device))

        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
        soft_labels = (100.0 * image_features_norm @ text_features_norm.T).softmax(dim=-1)

        predictions = torch.argmax(soft_labels, dim=-1)
        acc = compute_accuracy(predictions, label_targets)
        acc = reduce_tensor(acc)
        max_acc_meter.update(acc.item(), len(label_targets) * dist.get_world_size())
        if dist.get_rank() == 0:
            tbar.set_description(f"Target Testing  [ MM-Accuracy {max_acc_meter.val:.4f} ({max_acc_meter.avg:.4f}) ]")

if dist.get_rank() == 0:
    logger.info(f"Target Dataset [ MM-Accuracy {max_acc_meter.avg:.4f} ]")


# Train the model
for batch_idx, batch in enumerate(train_loader):
    loss_meter = AverageMeter()

    temp_dataset = UnlabeledDatasetV6(batch, extend=args.extend, class_names=class_names)
    temp_sampler = torch.utils.data.DistributedSampler(temp_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    temp_loader = DataLoader(temp_dataset, sampler=temp_sampler, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)

    if memory_bank.size() < args.max_len:
        compute_pseudo_labels(temp_dataset, adapt_transform, model_without_ddp, repeat=0, extend=args.extend)
    else:
        compute_pseudo_labels(temp_dataset, adapt_transform, model_without_ddp, repeat=args.repeat, extend=args.extend)

    if memory_bank.size() >= args.max_len:
        model.train()
        freeze_norm_layer(model_without_ddp)
        tbar = tqdm(range(args.iters * len(temp_loader)), dynamic_ncols=True) if dist.get_rank() == 0 else None
        for iter in range(args.iters):
            for images, pseudo_labels, soft_labels in temp_loader:
                optimizer.zero_grad()

                images = images.to(device)
                pseudo_labels = pseudo_labels.to(device)
                soft_labels = soft_labels.to(device)

                image_features = model_without_ddp.encode_image(images)
                text_features = model_without_ddp.encode_text(text_inputs.to(device))

                image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
                text_features_norm = text_features / text_features.norm(dim=-1, keepdim=True)
                outputs = 100.0 * image_features_norm @ text_features_norm.T

                loss = criterion(outputs, pseudo_labels)
                if args.pi > 0:
                    loss += F.kl_div(torch.log_softmax(outputs, dim=-1), soft_labels, reduction='batchmean')
                loss.backward()
                optimizer.step()

                loss_meter.update(loss.item(), images.size(0))
                if dist.get_rank() == 0:
                    tbar.set_description(f"Target Training [ Batch {batch_idx+1}/{len(train_loader)} | Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) ]")
                    tbar.update(1)
        
        if dist.get_rank() == 0: tbar.close()

        momentum(model_without_ddp, m=args.pi)

        if dist.get_rank() == 0 and args.save_model:
            torch.save(model_without_ddp.state_dict(), os.path.join(args.save_path, f"{experiment_name}_backbone_adapt.pth"))
            print(f"Save model to {args.save_path}")


# Test final model on the target dataset
model.eval()
mm_acc_meter = AverageMeter()

tbar = tqdm(train_loader, desc='Testing', dynamic_ncols=True) if dist.get_rank() == 0 else train_loader
for batch in tbar:
    temp_dataset = UnlabeledDatasetV6(batch, extend=args.extend, class_names=class_names, verbose=False)
    compute_pseudo_labels(temp_dataset, adapt_transform, model_without_ddp, repeat=0, verbose=False)

    if dist.get_rank() == 0:
        acc = temp_dataset.acc
        mm_acc_meter.update(acc, len(temp_dataset.samples))

if dist.get_rank() == 0:
    logger.info(f"Target Dataset [ Accuracy {mm_acc_meter.avg:.4f} ]"); print(f"Target Dataset [ Accuracy {mm_acc_meter.avg:.4f} ]")
