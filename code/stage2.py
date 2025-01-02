import json
import os
from pathlib import Path
import sys
import time
from typing import List, Tuple, Dict, Union, Optional
import warnings
import pandas as pd
import pickle
import torch
import scanpy as sc
import numpy as np
import wandb
from scipy.sparse import issparse
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchtext.vocab import Vocab
from torchtext._torchtext import (
    Vocab as VocabPybind,
)
from sklearn.metrics import confusion_matrix

sys.path.insert(0, "./")
import scgpt
from scgpt.model import TransformerModel, AdversarialDiscriminator
from scgpt.tokenizer import tokenize_and_pad_batch, random_mask_value
from scgpt.tokenizer.gene_tokenizer import GeneVocab
from scgpt.preprocess import Preprocessor
from scgpt import SubsetsBatchSampler
from scgpt.utils import set_seed, category_str2int, eval_scib_metrics
from model import FastKAN

sc.set_figure_params(figsize=(6, 6))
os.environ["KMP_WARNINGS"] = "off"
warnings.filterwarnings('ignore')



root = "./save/"
path = "muto_llm_finetuning_muto2021-xxxxx-xx-xx"

load_model_path = os.path.join(root, path)


hyperparameter_defaults = dict(
seed=0,
data_name="muto2021",
do_train=True,
load_model=load_model_path,
mask_ratio=0.0,
epochs=10,
n_bins=51,
MVC=False, # Masked value prediction for cell embedding
ecs_thres=0.0, # Elastic cell similarity objective, 0.0 to 1.0, 0.0 to disable
dab_weight=0.0,
lr=5e-5,       #1e-4,
batch_size=4,
layer_size=128,
nlayers=4,  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead=4,  # number of heads in nn.MultiheadAttention
dropout=0.2,  # dropout probability
schedule_ratio=0.9,  # ratio of epochs for learning rate schedule
save_eval_interval=5,
fast_transformer=True,
pre_norm=False,
amp=True,  # Automatic Mixed Precision
include_zero_gene = False,
freeze = True, #freeze
DSBN = False,  # Domain-spec batchnorm
)

run = wandb.init(
    config=hyperparameter_defaults,
    project="distillation_student_model_muto",
    reinit=True,
    settings=wandb.Settings(start_method="fork"),
)

config = wandb.config

# settings for input and preprocessing
pad_token = "<pad>"
special_tokens = [pad_token, "<cls>", "<eoc>"]
mask_ratio = config.mask_ratio
mask_value = "auto"  # for masked values, now it should always be auto

include_zero_gene = config.include_zero_gene  # if True, include zero genes among hvgs in the training
max_seq_len = 3001
n_bins = config.n_bins

# input/output representation
input_style = "binned"  # "normed_raw", "log1p", or "binned"
output_style = "binned"  # "normed_raw", "log1p", or "binned"

# settings for training
MLM = False  # whether to use masked language modeling, currently it is always on.
CLS = True  # celltype classification objective
ADV = False  # Adversarial training for batch correction
CCE = False  # Contrastive cell embedding objective
MVC = config.MVC  # Masked value prediction for cell embedding
ECS = config.ecs_thres > 0  # Elastic cell similarity objective
DAB = False  # Domain adaptation by reverse backpropagation, set to 2 for separate optimizer
INPUT_BATCH_LABELS = False  # TODO: have these help MLM and MVC, while not to classifier
input_emb_style = "continuous"  # "category" or "continuous" or "scaling"
cell_emb_style = "cls"  # "avg-pool" or "w-pool" or "cls"
adv_E_delay_epochs = 0  # delay adversarial training on encoder for a few epochs
adv_D_delay_epochs = 0
mvc_decoder_style = "inner product"
ecs_threshold = config.ecs_thres
dab_weight = config.dab_weight

explicit_zero_prob = MLM and include_zero_gene  # whether explicit bernoulli for zeros
do_sample_in_train = False and explicit_zero_prob  # sample the bernoulli in training

per_seq_batch_sample = False


# settings for the model
fast_transformer = config.fast_transformer
fast_transformer_backend = "flash"  # "linear" or "flash"
embsize = config.layer_size  # embedding dimension
d_hid = config.layer_size  # dimension of the feedforward network in TransformerEncoder
nlayers = config.nlayers  # number of TransformerEncoderLayer in TransformerEncoder
nhead = config.nhead  # number of heads in nn.MultiheadAttention
dropout = config.dropout  # dropout probability



assert input_style in ["normed_raw", "log1p", "binned"]
assert output_style in ["normed_raw", "log1p", "binned"]
assert input_emb_style in ["category", "continuous", "scaling"]
if input_style == "binned":
    if input_emb_style == "scaling":
        raise ValueError("input_emb_style `scaling` is not supported for binned input.")
elif input_style == "log1p" or input_style == "normed_raw":
    if input_emb_style == "category":
        raise ValueError(
            "input_emb_style `category` is not supported for log1p or normed_raw input."
        )

if input_emb_style == "category":
    mask_value = n_bins + 1
    pad_value = n_bins  # for padding gene expr values
    n_input_bins = n_bins + 2
else:
    mask_value = -1
    pad_value = -2
    n_input_bins = n_bins



data_name = config.data_name  


data_dir = Path(f"../data/{data_name}")
adata = sc.read(data_dir / f"train_data.h5ad")
adata_test = sc.read(data_dir / f"test_data.h5ad")
adata.obs["celltype"] = adata.obs["cell_type"].astype("category")
adata_test.obs["celltype"] = adata_test.obs["cell_type"].astype("category")
adata.obs["batch_id"]  = adata.obs["str_batch"] = "0"
adata_test.obs["batch_id"]  = adata_test.obs["str_batch"] = "1"    
data_is_raw = False
filter_gene_by_counts = False
adata_test_raw = adata_test.copy()
adata = adata.concatenate(adata_test, batch_key="str_batch")


sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3")
rna_hvg = adata[:, adata.var.highly_variable]
rna_hvg.layers["counts"] = rna_hvg.X.copy()
sc.pp.normalize_total(rna_hvg)
sc.pp.log1p(rna_hvg)
sc.pp.scale(rna_hvg)

sc.tl.pca(rna_hvg, n_comps=10, svd_solver="auto")
sc.pp.neighbors(rna_hvg, use_rep="X_pca", metric="cosine")
sc.tl.umap(rna_hvg)



# make the batch category column
batch_id_labels = adata.obs["str_batch"].astype("category").cat.codes.values
adata.obs["batch_id"] = batch_id_labels
celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
celltypes = adata.obs["celltype"].unique()
num_types = len(np.unique(celltype_id_labels))
id2type = dict(enumerate(adata.obs["celltype"].astype("category").cat.categories))
adata.obs["celltype_id"] = celltype_id_labels
adata.var["gene_name"] = adata.var.index.tolist()




save_dir = Path(f"./save/distillation_student_model_{data_name}-{time.strftime('%b%d-%H-%M')}/")
save_dir.mkdir(parents=True, exist_ok=True)
print(f"save to {save_dir}")


logger = scgpt.logger
scgpt.utils.add_file_handler(logger, save_dir / "run.log")


if config.load_model is not None:
    model_dir = Path(config.load_model)
    model_config_file = "../save/scGPT_human/args.json"
    model_file = model_dir / "model.pt"
    vocab_file = model_dir / "vocab.json"

    vocab = GeneVocab.from_file(vocab_file)
    # shutil.copy(vocab_file, save_dir / "vocab.json")
    for s in special_tokens:
        if s not in vocab:
            vocab.append_token(s)

    adata.var["id_in_vocab"] = [
        1 if gene in vocab else -1 for gene in adata.var["gene_name"]
    ]
    gene_ids_in_vocab = np.array(adata.var["id_in_vocab"])
    logger.info(
        f"match {np.sum(gene_ids_in_vocab >= 0)}/{len(gene_ids_in_vocab)} genes "
        f"in vocabulary of size {len(vocab)}."
    )
    adata = adata[:, adata.var["id_in_vocab"] >= 0]

    # model
    with open(model_config_file, "r") as f:
        model_configs = json.load(f)
    logger.info(
        f"Resume model from {model_file}, the model args will override the "
        f"config {model_config_file}."
    )
    embsize = model_configs["embsize"]
    nhead = model_configs["nheads"]
    d_hid = model_configs["d_hid"]
    nlayers = model_configs["nlayers"]
    n_layers_cls = model_configs["n_layers_cls"]



preprocessor = Preprocessor(
        use_key="X",  # the key in adata.layers to use as raw data
        filter_gene_by_counts=filter_gene_by_counts,  # step 1
        filter_cell_by_counts=True, # Flase  # step 2
        normalize_total=1e4,  # 3. whether to normalize the raw data and to what sum
        result_normed_key="X_normed",  # the key in adata.layers to store the normalized data
        log1p=data_is_raw,  # 4. whether to log1p the normalized data
        result_log1p_key="X_log1p",
        subset_hvg=False,  # 5. whether to subset the raw data to highly variable genes
        hvg_flavor="seurat_v3" if data_is_raw else "cell_ranger",
        binning=n_bins,  # 6. whether to bin the raw data and to what number of bins
        result_binned_key="X_binned",  # the key in adata.layers to store the binned data
    )



adata_test = adata[adata.obs["str_batch"] == "1"]
adata = adata[adata.obs["str_batch"] == "0"]

rna_hvg_train = rna_hvg[rna_hvg.obs["str_batch"] == "0"]
rna_hvg_test = rna_hvg[rna_hvg.obs["str_batch"] == "1"]

preprocessor(adata, batch_key=None)
preprocessor(adata_test, batch_key=None)



input_layer_key = {  # the values of this map coorespond to the keys in preprocessing
    "normed_raw": "X_normed",
    "log1p": "X_normed",
    "binned": "X_binned",
}[input_style]
all_counts = (
    adata.layers[input_layer_key].A
    if issparse(adata.layers[input_layer_key])
    else adata.layers[input_layer_key]
)
genes = adata.var["gene_name"].tolist()

celltypes_labels = adata.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata.obs["batch_id"].tolist()
num_batch_types = len(set(batch_ids))
batch_ids = np.array(batch_ids)


if config.load_model is None:
    vocab = Vocab(
        VocabPybind(genes + special_tokens, None)
    )  # bidirectional lookup [gene <-> int]
vocab.set_default_index(vocab["<pad>"])
gene_ids = np.array(vocab(genes), dtype=int)


(
    feature_train,
    feature_valid,
    train_data,
    valid_data,
    train_celltype_labels,
    valid_celltype_labels,
    train_batch_labels,
    valid_batch_labels,
) = train_test_split(
    rna_hvg_train.X, all_counts, celltypes_labels, batch_ids, test_size=1/8, shuffle=True
)


tokenized_train = tokenize_and_pad_batch(
    train_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)
tokenized_valid = tokenize_and_pad_batch(
    valid_data,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,
    include_zero_gene=include_zero_gene,
)

ntokens = len(vocab)  # size of vocabulary

model = TransformerModel(
    ntokens,
    embsize,
    nhead,
    d_hid,
    nlayers,
    nlayers_cls=3,
    n_cls=num_types if CLS else 1,
    vocab=vocab,
    dropout=dropout,
    pad_token=pad_token,
    pad_value=pad_value,
    do_mvc=MVC,
    do_dab=DAB,
    use_batch_labels=INPUT_BATCH_LABELS,
    num_batch_labels=num_batch_types,
    domain_spec_batchnorm=config.DSBN,
    input_emb_style=input_emb_style,
    n_input_bins=n_input_bins,
    cell_emb_style=cell_emb_style,
    mvc_decoder_style=mvc_decoder_style,
    ecs_threshold=ecs_threshold,
    explicit_zero_prob=explicit_zero_prob,
    use_fast_transformer=fast_transformer,
    fast_transformer_backend=fast_transformer_backend,
    pre_norm=config.pre_norm,
)


if config.load_model is not None:
    try:
        model.load_state_dict(torch.load(model_file))
        logger.info(f"Loading all model params from {model_file}")
    except:
        # only load params that are in the model and match the size
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_file)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        # for k, v in pretrained_dict.items():
        #     logger.info(f"Loading params {k} with shape {v.shape}")
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)



model.eval()

if config.freeze:
    for param in model.parameters():
        param.requires_grad = False


# calculate the number of parameters that require grad

n_params = 0

for p in model.parameters():
    if p.requires_grad:
        n_params += p.numel()

print(f"number of parameters that require grad: {n_params}")



def prepare_data(sort_seq_batch=False) -> Tuple[Dict[str, torch.Tensor]]:
    masked_values_train = random_mask_value(
        tokenized_train["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    masked_values_valid = random_mask_value(
        tokenized_valid["values"],
        mask_ratio=mask_ratio,
        mask_value=mask_value,
        pad_value=pad_value,
    )
    print(
        f"random masking at epoch {epoch:3d}, ratio of masked values in train: ",
        f"{(masked_values_train == mask_value).sum() / (masked_values_train - pad_value).count_nonzero():.4f}",
    )

    input_gene_ids_train, input_gene_ids_valid = (
        tokenized_train["genes"],
        tokenized_valid["genes"],
    )
    input_values_train, input_values_valid = masked_values_train, masked_values_valid
    target_values_train, target_values_valid = (
        tokenized_train["values"],
        tokenized_valid["values"],
    )

    tensor_batch_labels_train = torch.from_numpy(train_batch_labels).long()
    tensor_batch_labels_valid = torch.from_numpy(valid_batch_labels).long()

    tensor_celltype_labels_train = torch.from_numpy(train_celltype_labels).long()
    tensor_celltype_labels_valid = torch.from_numpy(valid_celltype_labels).long()


    tensor_feature_train = torch.from_numpy(feature_train).float()
    tensor_feature_valid = torch.from_numpy(feature_valid).float()

    if sort_seq_batch:  # TODO: update to random pick seq source in each traning batch
        train_sort_ids = np.argsort(train_batch_labels)
        input_gene_ids_train = input_gene_ids_train[train_sort_ids]
        input_values_train = input_values_train[train_sort_ids]
        target_values_train = target_values_train[train_sort_ids]
        tensor_batch_labels_train = tensor_batch_labels_train[train_sort_ids]
        tensor_celltype_labels_train = tensor_celltype_labels_train[train_sort_ids]

        valid_sort_ids = np.argsort(valid_batch_labels)
        input_gene_ids_valid = input_gene_ids_valid[valid_sort_ids]
        input_values_valid = input_values_valid[valid_sort_ids]
        target_values_valid = target_values_valid[valid_sort_ids]
        tensor_batch_labels_valid = tensor_batch_labels_valid[valid_sort_ids]
        tensor_celltype_labels_valid = tensor_celltype_labels_valid[valid_sort_ids]

    train_data_pt = {
        "gene_ids": input_gene_ids_train,
        "values": input_values_train,
        "target_values": target_values_train,
        "batch_labels": tensor_batch_labels_train,
        "celltype_labels": tensor_celltype_labels_train,
        "features": tensor_feature_train,
    }
    valid_data_pt = {
        "gene_ids": input_gene_ids_valid,
        "values": input_values_valid,
        "target_values": target_values_valid,
        "batch_labels": tensor_batch_labels_valid,
        "celltype_labels": tensor_celltype_labels_valid,
        "features": tensor_feature_valid
    }

    return train_data_pt, valid_data_pt


# dataset
class SeqDataset(Dataset):
    def __init__(self, data: Dict[str, torch.Tensor]):
        self.data = data

    def __len__(self):
        return self.data["gene_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# data_loader
def prepare_dataloader(
    data_pt: Dict[str, torch.Tensor],
    batch_size: int,
    shuffle: bool = False,
    intra_domain_shuffle: bool = False,
    drop_last: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    if num_workers == 0:
        num_workers = min(len(os.sched_getaffinity(0)), batch_size // 2)

    dataset = SeqDataset(data_pt)

    if per_seq_batch_sample:
        # find the indices of samples in each seq batch
        subsets = []
        batch_labels_array = data_pt["batch_labels"].numpy()
        for batch_label in np.unique(batch_labels_array):
            batch_indices = np.where(batch_labels_array == batch_label)[0].tolist()
            subsets.append(batch_indices)
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=SubsetsBatchSampler(
                subsets,
                batch_size,
                intra_subset_shuffle=intra_domain_shuffle,
                inter_subset_shuffle=shuffle,
                drop_last=drop_last,
            ),
            num_workers=num_workers,
            pin_memory=True,
        )
        return data_loader

    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
    )
    return data_loader

batch_size = 8
eval_batch_size = 8

tensor_test = torch.from_numpy(rna_hvg_test.X).float()

all_counts = (
    adata_test.layers[input_layer_key].A
    if issparse(adata_test.layers[input_layer_key])
    else adata_test.layers[input_layer_key]
)

celltypes_labels = adata_test.obs["celltype_id"].tolist()  # make sure count from 0
celltypes_labels = np.array(celltypes_labels)

batch_ids = adata_test.obs["batch_id"].tolist()
batch_ids = np.array(batch_ids)

tokenized_test = tokenize_and_pad_batch(
    all_counts,
    gene_ids,
    max_len=max_seq_len,
    vocab=vocab,
    pad_token=pad_token,
    pad_value=pad_value,
    append_cls=True,  # append <cls> token at the beginning
    include_zero_gene=include_zero_gene,
)

input_values_test = random_mask_value(
    tokenized_test["values"],
    mask_ratio=mask_ratio,
    mask_value=mask_value,
    pad_value=pad_value,
)

test_data_pt = {
    "gene_ids": tokenized_test["genes"],
    "values": input_values_test,
    "target_values": tokenized_test["values"],
    "batch_labels": torch.from_numpy(batch_ids).long(),
    "celltype_labels": torch.from_numpy(celltypes_labels).long(),
    "features": tensor_test,
}

test_loader = prepare_dataloader(
    test_data_pt,
    batch_size=eval_batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
)

epoch = 1

train_data_pt, valid_data_pt = prepare_data(sort_seq_batch=per_seq_batch_sample)
train_loader = prepare_dataloader(
    train_data_pt,
    batch_size=batch_size,
    shuffle=False,
    intra_domain_shuffle=True,
    drop_last=False,
)
valid_loader = prepare_dataloader(
    valid_data_pt,
    batch_size=eval_batch_size,
    shuffle=False,
    intra_domain_shuffle=False,
    drop_last=False,
)


kan_model = FastKAN([2000, 64, num_types])

class BaseLoss:
    eps = 1e-9
    def __init__(self, model):
        self.n_output = len(list(model.clusters[0].parameters())[0])
        self.weight = 1
    @staticmethod
    def compute_distance(is_binary_input, output, target):
        """\
        Compute the distance between target and output with BCE if binary data or MSE for all others.
        """
        if is_binary_input:
            return F.binary_cross_entropy(output, target)
        else:
            return F.mse_loss(output, target)


class SelfEntropyLoss(BaseLoss):
    """
    Entropy regularization to prevent trivial solution.
    """
    def __init__(self, loss_weight):
        # super().__init__()
        self.prob_layer = torch.nn.Softmax(dim=1)
        self.weight = loss_weight

    def __call__(self, cluster_outputs):
        loss = 0.
        eps = 1e-9
        cluster_outputs = self.prob_layer(cluster_outputs)
        prob_mean = cluster_outputs.mean(dim=0)
        prob_mean[(prob_mean < eps).data] = eps
        # print(prob_mean)
        # print(torch.log(prob_mean))
        loss = -(prob_mean * torch.log(prob_mean)).sum()

        loss *= self.weight
        return loss
    
class DDCLoss(BaseLoss):
    def __init__(self, n_output, loss_weight):
        # super().__init__()
        self.weight = loss_weight
        self.n_output = n_output
        self.eye = torch.eye(self.n_output, device=torch.device("cuda"))
        self.prob_layer = torch.nn.Softmax(dim=1)

    @staticmethod
    def triu(X):
        """\ 
        Sum of strictly upper triangular part.
        """
        return torch.sum(torch.triu(X, diagonal=1))

    @staticmethod
    def _atleast_epsilon(X, eps=1e-9):
        """
        Ensure that all elements are >= `eps`.
        """
        return torch.where(X < eps, X.new_tensor(eps), X)

    @staticmethod
    def d_cs(A, K, n_clusters):
        """
        Cauchy-Schwarz divergence.
        """
        nom = torch.t(A) @ K @ A
        dnom_squared = torch.unsqueeze(torch.diagonal(nom), -1) @ torch.unsqueeze(
            torch.diagonal(nom), 0
        )

        nom = DDCLoss._atleast_epsilon(nom)
        dnom_squared = DDCLoss._atleast_epsilon(dnom_squared, eps=BaseLoss.eps ** 2)

        d = (
            2
            / (n_clusters * (n_clusters - 1))
            * DDCLoss.triu(nom / torch.sqrt(dnom_squared))
        )
        return d

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k

    @staticmethod
    def kernel_from_distance_matrix(dist, rel_sigma, min_sigma=BaseLoss.eps):
        """\
        Compute a Gaussian kernel matrix from a distance matrix.
        """
        # `dist` can sometimes contain negative values due to floating point errors, so just set these to zero.
        dist = F.relu(dist)
        sigma2 = rel_sigma * torch.median(dist)
        # Disable gradient for sigma
        sigma2 = sigma2.detach()
        sigma2 = torch.where(sigma2 < min_sigma, sigma2.new_tensor(min_sigma), sigma2)
        k = torch.exp(-dist / (2 * sigma2))
        return k

    @staticmethod
    def cdist(X, Y):
        """\
        Pairwise distance between rows of X and rows of Y.
        """
        xyT = X @ torch.t(Y)
        x2 = torch.sum(X ** 2, dim=1, keepdim=True)
        y2 = torch.sum(Y ** 2, dim=1, keepdim=True)
        d = x2 - 2 * xyT + torch.t(y2)
        return d

    @staticmethod
    def vector_kernel(x, rel_sigma=0.15):
        """\
        Compute a kernel matrix from the rows of a matrix.
        """
        return DDCLoss.kernel_from_distance_matrix(DDCLoss.cdist(x, x), rel_sigma)

    def __call__(self, hidden, cluster_outputs):
        loss = 0.

        cluster_outputs = self.prob_layer(cluster_outputs)
        hidden_kernel = DDCLoss.vector_kernel(hidden)
        # L_1 loss
        loss = DDCLoss.d_cs(cluster_outputs, hidden_kernel, self.n_output)

        # L_3 loss
        m = torch.exp(-DDCLoss.cdist(cluster_outputs, self.eye))
        loss += DDCLoss.d_cs(m, hidden_kernel, self.n_output)
        loss *= self.weight

        return loss


import torch.optim as optim

from tqdm import tqdm

class PyTorchDistiller:
    def __init__(self, student, teacher, loss_weights=[0.5, 0.5]):
        self.student = student
        self.teacher = teacher
        self.teacher.eval()  # Set teacher to evaluation mode
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.student.parameters(),lr = 5e-4)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.student.to(self.device)
        self.teacher.to(self.device)
        self.random_seed = 0

        self.sce = SelfEntropyLoss(loss_weights[0])
        self.ddc = DDCLoss(student.n_classes, loss_weights[1])

    def distillation_loss(self, student_logits, teacher_logits, temperature,
                        alpha):
        soft_loss = nn.KLDivLoss()(
            nn.functional.log_softmax(student_logits / temperature, dim=1),
            nn.functional.softmax(teacher_logits / temperature, dim=1))
        return alpha * soft_loss

    def fit(self, train_loader, valid_loader, epochs=10):
        GLOBAL_SEED = self.random_seed
        set_seed(GLOBAL_SEED)


        best_eval_acc = 0


        device = 'cuda'
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(device)            
        for epoch in range(epochs):
            train_loss, train_acc = self.train_one_epoch(data_loader=train_loader,
                                                        epoch=epoch)
            val_acc = self.evaluate(data_loader=valid_loader,
                                            epoch=epoch)
            
            # log loss and accuracy to wandb
            wandb.log({"train_loss": train_loss, "train_acc": train_acc, "val_acc": val_acc})
            
            if val_acc > best_eval_acc:
                print("the acc in epoch {} is {}, better than the best acc {}".format(epoch, val_acc, best_eval_acc))
                best_eval_acc = val_acc
                torch.save(self.student.state_dict(), save_dir / 'best_model.pth')
            else:
                print("the acc in epoch {} is {}, not better than the best acc {}".format(epoch, val_acc, best_eval_acc))
        print('Training finished!')

        print('Testing...')
        self.student.load_state_dict(torch.load(save_dir / 'best_model.pth'))
        print("loading the best eval model, which has the acc of ", best_eval_acc)

        result = self.test(test_loader)
        return result

    def train_one_epoch(self, data_loader, epoch, alpha=0.1, temperature=3):
        self.student.train()
        self.optimizer.zero_grad()
        accu_loss = torch.zeros(1).to(self.device)
        accu_loss_sce = torch.zeros(1).to(self.device)
        accu_loss_ddc = torch.zeros(1).to(self.device)

        accu_num = torch.zeros(1).to(self.device)
        sample_num = 0
        for step, batch_data in enumerate(data_loader):

            input_gene_ids = batch_data["gene_ids"].to(self.device)
            input_values = batch_data["values"].to(self.device)
            target_values = batch_data["target_values"].to(self.device)
            batch_labels = batch_data["batch_labels"].to(self.device)
            celltype_labels = batch_data["celltype_labels"].to(self.device)
            src_key_padding_mask = input_gene_ids.eq(vocab[pad_token])
            feature_data = batch_data["features"].to(self.device)
            self.optimizer.zero_grad()
            sample_num += feature_data.shape[0]
            hidden, student_outputs = self.student(feature_data)

            output_dict = self.teacher(
                input_gene_ids,
                input_values,
                src_key_padding_mask=src_key_padding_mask,
                batch_labels=batch_labels if INPUT_BATCH_LABELS or config.DSBN else None,
                CLS=CLS,
                CCE=CCE,
                MVC=MVC,
                ECS=ECS,
                do_sample=do_sample_in_train,
                #generative_training=False
            )
            teacher_outputs = output_dict['cls_output'].detach(
            )  # Detach teacher outputs so no gradients are backpropagated



            # teacher_outputs = output_dict
            pred_classes = torch.max(student_outputs, dim=1)[1]
            accu_num += torch.eq(pred_classes, celltype_labels).sum()

            loss_dist = self.criterion(
                student_outputs, celltype_labels) + self.distillation_loss(
                    student_outputs, teacher_outputs, temperature, alpha)
            
            loss_sce = self.sce(student_outputs)
            loss_ddc = self.ddc(hidden, student_outputs)

            loss = loss_dist + loss_sce + loss_ddc

            loss.backward()
            nn.utils.clip_grad_norm_(self.student.parameters(), 25)

            accu_loss += loss.detach()
            accu_loss_sce += loss_sce.detach()
            accu_loss_ddc += loss_ddc.detach()

            if (step + 1) % 100 == 0:
                print("[train epoch {}] | [step {} - total {}] loss: {:.3f}, sce loss: {:.3f}, ddc loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                                step + 1,
                                                                                len(data_loader),
                                                                                accu_loss.item() / (step + 1),
                                                                                accu_loss_sce.item() / (step + 1),
                                                                                accu_loss_ddc.item() / (step + 1),
                                                                                accu_num.item() / sample_num))
            if not torch.isfinite(loss):
                print('WARNING: non-finite loss, ending training ', loss)
                sys.exit(1)
            self.optimizer.step()
        return accu_loss.item() / (step + 1), accu_num.item() / sample_num       
    
    @torch.no_grad()
    def evaluate(self, data_loader, epoch):
        self.student.eval()
        accu_num = torch.zeros(1).to(self.device)
        sample_num = 0
        for step, batch_data in enumerate(data_loader):

            celltype_labels = batch_data["celltype_labels"].to(self.device)
            feature_data = batch_data["features"].to(self.device)
            sample_num += feature_data.shape[0]
            _, pred = self.student(feature_data)
            pred_classes = torch.max(pred, dim=1)[1]
            accu_num += torch.eq(pred_classes, celltype_labels.to(self.device)).sum()


            if (step + 1) % 100 == 0:
                print("[valid epoch {}] | [step {} - total {}] acc: {:.3f}".format(epoch, step + 1, len(data_loader), accu_num.item() / sample_num))
        return accu_num.item() / sample_num

    @torch.no_grad()
    def test(self,test_loader: DataLoader) -> float:

        self.student.eval()
        predictions = []
        labels = []
        for step, batch_data in enumerate(test_loader):

            celltype_labels = batch_data["celltype_labels"].to(self.device)
            feature_data = batch_data["features"].to(self.device)
            labels.append(celltype_labels)
            _, pred = self.student(feature_data)
            pred_classes = torch.max(pred, dim=1)[1]
            predictions.append(pred_classes)

        predictions = torch.cat(predictions).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average="macro")
        recall = recall_score(labels, predictions, average="macro")
        macro_f1 = f1_score(labels, predictions, average="macro")

        logger.info(
            f"Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, "
            f"Macro F1: {macro_f1:.3f}"
        )

        results = {
            "test/accuracy": accuracy,
            "test/precision": precision,
            "test/recall": recall,
            "test/macro_f1": macro_f1,
        }

        return predictions, celltypes_labels, results


distiller = PyTorchDistiller(student=kan_model, teacher=model)        

predictions, celltypes_labels, results = distiller.fit(train_loader=train_loader, valid_loader=valid_loader, epochs=10)

save_dict = {
"predictions": predictions,
"labels": celltypes_labels,
"results": results,
"id_maps": id2type
}
with open(save_dir / "results.pkl", "wb") as f:
    pickle.dump(save_dict, f)

wandb.log(results)