import numpy as np
import torch.nn as nn
import random
import torch
from torch import device
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import TensorDataset

from multi_graph_gen import clf_graph
import pandas as pd
import scipy.stats as stats
# import os
import dgl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
import copy
import csv as _csv

import warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")

inputfolder = "data/processed_data_discovery/"
outputfolder = "data/ImmGraph_results/"


# R2损失函数
class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_true, y_pred):
        ss_total = torch.sum((y_true - torch.mean(y_true)) ** 2)
        ss_residual = torch.sum((y_true - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return 1 - r2  # minimize this value


def load_configuratiton(model, device):
    config = dict()
    config['device'] = device
    config['model_name'] = 'rlmodel'
    config['optimizer'] = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)                    
    config['scheduler'] = ReduceLROnPlateau(config['optimizer'], 'min', patience=3, factor=0.5, min_lr=1e-6)
    config['num_epochs'] = 100
    config['save_path'] = "E:/Multi-omic Immunity/GCN_immune/output_lgg/best_model.pth"
    config['early_stopping'] = 20
    config['clip'] = 1.0

    return config


# Examine data distribtion
def no_difference(dataset1, dataset2):
    reoccur = np.mean(dataset1.VITAL_STATUS) - np.mean(dataset2.VITAL_STATUS)
    OS = stats.mannwhitneyu(dataset1.OS.tolist(), dataset2.OS.tolist())
    OSmin = min(dataset1.OS.tolist()) - min(dataset2.OS.tolist())
    OSmax = max(dataset1.OS.tolist()) - max(dataset2.OS.tolist())

    if OS.pvalue < 0.05 or abs(OSmin) > 200 or abs(OSmax) > 300 or abs(reoccur) > 0.3:
        return False
    else:
        return True

# set random seed
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# set train dataloder, calculate zscore
def train_datatotensor(LIST, nodes_dna, nodes_rna, nodes_protein, edges_dna,
                       edges_rna, edges_protein, edges_dnarna, edges_rnapro, label):
    # How many nodes are used
    nodes_dna = nodes_dna.iloc[:, LIST]
    nodes_rna = nodes_rna.iloc[:, LIST]
    nodes_protein = nodes_protein.iloc[:, LIST]

    imrna_label = label.iloc[LIST].loc[:, ['RNA_PC1', 'RNA_PC2']].values
    imrna_label = torch.tensor(imrna_label, dtype=torch.float32)

    imdna_label = label.iloc[LIST].loc[:, ['Mutation_PC1', 'Mutation_PC2']].values
    imdna_label = torch.tensor(imdna_label, dtype=torch.float32)

    impro_label = label.iloc[LIST].loc[:, ['Protein_PC1', 'Protein_PC2']].values
    impro_label = torch.tensor(impro_label, dtype=torch.float32)

    srcdna_id = torch.tensor(edges_dna.loc[:, 'srcdna_id'].values)
    dstdna_id = torch.tensor(edges_dna.loc[:, 'dstdna_id'].values)
    srcrna_id = torch.tensor(edges_rna.loc[:, 'srcrna_id'].values)
    dstrna_id = torch.tensor(edges_rna.loc[:, 'dstrna_id'].values)
    srcpro_id = torch.tensor(edges_protein.loc[:, 'srcpro_id'].values)
    dstpro_id = torch.tensor(edges_protein.loc[:, 'dstpro_id'].values)

    dna_id = torch.tensor(edges_dnarna.loc[:, 'dna_id'].values)
    rna1_id = torch.tensor(edges_dnarna.loc[:, 'rna1_id'].values)

    rna_id = torch.tensor(edges_rnapro.loc[:, 'rna_id'].values)
    pro_id = torch.tensor(edges_rnapro.loc[:, 'pro_id'].values)

    nodes_dna = torch.from_numpy(nodes_dna.values).float()
    nodes_rna = torch.from_numpy(nodes_rna.values).float()
    nodes_protein = torch.from_numpy(nodes_protein.values).float()

    #print(f"nodes_dna size: {nodes_dna.size(0)}")
    #print(f"nodes_rna size: {nodes_rna.size(0)}")
    #print(f"nodes_protein size: {nodes_protein.size(0)}")
    #print(f"imrna_label size: {imrna_label.size(0)}")
    return nodes_dna, nodes_rna, nodes_protein, srcdna_id, dstdna_id, srcrna_id, dstrna_id, srcpro_id, dstpro_id, dna_id, rna1_id, rna_id, pro_id, imdna_label, imrna_label, impro_label

def calculate_metrics(predictions, targets):
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    mae = mean_absolute_error(targets, predictions)
    rmse = np.sqrt(mean_squared_error(targets, predictions))
    r2 = r2_score(targets, predictions)
    return mae, rmse, r2
    
def compute_all_metrics(pred, target):
    
    y_pred = pred.detach().cpu().numpy().reshape(-1)
    y_true = target.detach().cpu().numpy().reshape(-1)

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    pearson = np.nan
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = stats.pearsonr(y_true, y_pred)[0]
    return {"MAE": mae, "MedAE": medae, "RMSE": rmse, "R2": r2, "Pearson": pearson}


class Regularization(nn.Module):
    def __init__(self, order, weight_decay):
        super(Regularization, self).__init__()
        self.order = order
        self.weight_decay = weight_decay

    def forward(self, model):
        reg_loss = 0
        for name, w in model.named_parameters():
            if 'weight' in name:
                reg_loss += torch.norm(w, p=self.order)
        reg_loss *= self.weight_decay
        return reg_loss


class MSEWithRegularization(nn.Module):
    def __init__(self, order=2, weight_decay=0.001):
        super(MSEWithRegularization, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.reg = Regularization(order=order, weight_decay=weight_decay)
    def forward(self, predictions, targets, model):
        # print("Predictions shape:", predictions.shape)
        # print("Targets shape:", targets.shape)
        mse_loss = self.mse_loss(predictions, targets)
        reg_loss = self.reg(model)
        total_loss = mse_loss + reg_loss
        return total_loss

def batched_predict(model, graph, batch_size, device):
    model.eval()
    with torch.no_grad():
        num_patients = graph.nodes['dna'].data['feat'].shape[1]
        patient_ids = torch.arange(num_patients, device=device)

        all_preds_dna = []
        all_preds_rna = []
        all_preds_pro = []

        output_dna, output_rna, output_pro, *_ = model(graph)

        for i in range(0, num_patients, batch_size):
            batch_ids = patient_ids[i:i+batch_size]

            pred_dna = output_dna[batch_ids]
            pred_rna = output_rna[batch_ids]
            pred_pro = output_pro[batch_ids]

            all_preds_dna.append(pred_dna)
            all_preds_rna.append(pred_rna)
            all_preds_pro.append(pred_pro)

        all_preds_dna = torch.cat(all_preds_dna, dim=0)
        all_preds_rna = torch.cat(all_preds_rna, dim=0)
        all_preds_pro = torch.cat(all_preds_pro, dim=0)

    return all_preds_dna, all_preds_rna, all_preds_pro


def check_for_nan(tensor, name="Tensor", epoch=None, batch_idx=None):
    if torch.isnan(tensor).any():
        print(f"Warning: {name} contains NaN values at Epoch {epoch}, Batch {batch_idx}")
        return True
    return False

def save_metrics_to_csv(metrics, fold, seed):
    # 定义 CSV 文件路径
    metrics_csv = f"fold_{fold}_metrics_seed_{seed}.csv"
    # 写入标题行
    header = ["Fold", "Seed", "MAE", "MedAE", "RMSE", "R2", "Pearson"]
    with open(metrics_csv, mode='a', newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:
            writer.writerow(header)
        # Write the metric data for each fold
        writer.writerow(metrics)

def compute_all_metrics(pred, target):
    y_pred = pred.detach().cpu().numpy().reshape(-1)
    y_true = target.detach().cpu().numpy().reshape(-1)
    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    pearson = np.nan
    if np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson = stats.pearsonr(y_true, y_pred)[0]
    return {"MAE": mae, "MedAE": medae, "RMSE": rmse, "R2": r2, "Pearson": pearson}

def train(train_loader, valid_loader, fold, seed, device, im, train_list, valid_list):
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = R2Loss()
    best_train = 0
    valid_rna_before = -float("inf")  # Initialize the worst R² value
    valid_dna_before = -float("inf")
    valid_pro_before = -float("inf")
    best_epoch = 0
    no_improvement_counter = 0  # Counter for consecutive non-improvements.
    patience = 10  # Maximum tolerance epoch count for no improvement
    loss_before = 10
    valid_rna_losses = []
    # print("train_loader shape:", train_loader.shape)
    initial_epochs = 150
    # for batch in train_loader:
    # print(batch)
    train_nodes_dna, train_nodes_rna, train_nodes_protein, train_srcdna_id, train_dstdna_id, train_srcrna_id, train_dstrna_id, train_srcpro_id, train_dstpro_id, train_dna_id, train_rna1_id, train_rna_id, train_pro_id, train_imdna, train_imrna, train_impro = train_loader
    # print("train_loader shape:", train_loader.shape)
    valid_nodes_dna, valid_nodes_rna, valid_nodes_protein, valid_srcdna_id, valid_dstdna_id, valid_srcrna_id, valid_dstrna_id, valid_srcpro_id, valid_dstpro_id, valid_dna_id, valid_rna1_id, valid_rna_id, valid_pro_id, valid_imdna, valid_imrna, valid_impro = valid_loader
    
    '''print("train edges dna:", len(train_srcdna_id))
    print("train edges rna:", len(train_srcrna_id))
    print("train edges pro:", len(train_srcpro_id))
    print("train edges dnarna:", len(train_dna_id))
    print("train edges rnapro:", len(train_rna_id))
    
    print("valid edges dna:", len(valid_srcdna_id))
    print("valid edges rna:", len(valid_srcrna_id))
    print("valid edges pro:", len(valid_srcpro_id))
    print("valid edges dnarna:", len(valid_dna_id))
    print("valid edges rnapro:", len(valid_rna_id))'''

    train_nodes_dna = train_nodes_dna.to(device)
    train_nodes_rna = train_nodes_rna.to(device)
    train_nodes_protein = train_nodes_protein.to(device)

    train_srcdna_id = train_srcdna_id.to(device)
    train_dstdna_id = train_dstdna_id.to(device)
    train_srcrna_id = train_srcrna_id.to(device)
    train_dstrna_id = train_dstrna_id.to(device)
    train_srcpro_id = train_srcpro_id.to(device)
    train_dstpro_id = train_dstpro_id.to(device)

    train_dna_id = train_dna_id.to(device)
    train_rna1_id = train_rna1_id.to(device)
    train_rna_id = train_rna_id.to(device)
    train_pro_id = train_pro_id.to(device)
    train_imrna = train_imrna.to(device)
    train_imdna = train_imdna.to(device)
    train_impro = train_impro.to(device)

    train_data_dict = {
        ('dna', 'dna_interact', 'dna'): (train_srcdna_id, train_dstdna_id),
        ('rna', 'rna_interact', 'rna'): (train_srcrna_id, train_dstrna_id),
        ('protein', 'pro_interact', 'protein'): (train_srcpro_id, train_dstpro_id),
        ('dna', 'transcribe', 'rna'): (train_dna_id, train_rna1_id),
        ('rna', 'translate', 'protein'): (train_rna_id, train_pro_id)}

    train_graph = dgl.heterograph(train_data_dict).to(device)  ##模型使用
    train_graph.nodes['dna'].data['feat'] = train_nodes_dna.to(device)
    train_graph.nodes['rna'].data['feat'] = train_nodes_rna.to(device)
    train_graph.nodes['protein'].data['feat'] = train_nodes_protein.to(device)
    
    '''print("==== DEVICE CHECK ====")
    print("train_nodes_dna.device:", train_nodes_dna.device)
    print("train_graph.device:", train_graph.device)
    print("graph dna feat device:", train_graph.nodes['dna'].data['feat'].device)'''
    
    '''print("dna nodes:", train_graph.num_nodes('dna'))
    print("max srcdna_id:", train_srcdna_id.max().item())
    print("max dstdna_id:", train_dstdna_id.max().item())

    print("rna nodes:", train_graph.num_nodes('rna'))
    print("max srcrna_id:", train_srcrna_id.max().item())
    print("max dstrna_id:", train_dstrna_id.max().item())

    print("protein nodes:", train_graph.num_nodes('protein'))
    print("max srcpro_id:", train_srcpro_id.max().item())
    print("max dstpro_id:", train_dstpro_id.max().item())'''

    # 添加病人编号到节点数据
    train_graph.nodes['dna'].data['patient_id'] = train_nodes_dna[:, -1]  # 假设病人编号在最后一列
    train_graph.nodes['rna'].data['patient_id'] = train_nodes_rna[:, -1]
    train_graph.nodes['protein'].data['patient_id'] = train_nodes_protein[:, -1]

    valid_nodes_dna = valid_nodes_dna.to(device)
    valid_nodes_rna = valid_nodes_rna.to(device)
    valid_nodes_protein = valid_nodes_protein.to(device)

    valid_srcdna_id = valid_srcdna_id.to(device)
    valid_dstdna_id = valid_dstdna_id.to(device)
    valid_srcrna_id = valid_srcrna_id.to(device)
    valid_dstrna_id = valid_dstrna_id.to(device)
    valid_srcpro_id = valid_srcpro_id.to(device)
    valid_dstpro_id = valid_dstpro_id.to(device)

    valid_dna_id = valid_dna_id.to(device)
    valid_rna1_id = valid_rna1_id.to(device)
    valid_rna_id = valid_rna_id.to(device)
    valid_pro_id = valid_pro_id.to(device)

    valid_imdna = valid_imdna.to(device)
    valid_imrna = valid_imrna.to(device)
    valid_impro = valid_impro.to(device)

    valid_data_dict = {
        ('dna', 'dna_interact', 'dna'): (valid_srcdna_id, valid_dstdna_id),
        ('rna', 'rna_interact', 'rna'): (valid_srcrna_id, valid_dstrna_id),
        ('protein', 'pro_interact', 'protein'): (valid_srcpro_id, valid_dstpro_id),
        ('dna', 'transcribe', 'rna'): (valid_dna_id, valid_rna1_id),
        ('rna', 'translate', 'protein'): (valid_rna_id, valid_pro_id)}

    valid_graph = dgl.heterograph(valid_data_dict).to(device)
    valid_graph.nodes['dna'].data['feat'] = valid_nodes_dna.to(device)
    valid_graph.nodes['rna'].data['feat'] = valid_nodes_rna.to(device)
    valid_graph.nodes['protein'].data['feat'] = valid_nodes_protein.to(device)
    
    '''print("valid_nodes_dna.device:", valid_nodes_dna.device)
    print("valid_graph.device:", valid_graph.device)
    print("valid dna feat device:", valid_graph.nodes['dna'].data['feat'].device)'''


    # 添加病人编号到节点数据
    valid_graph.nodes['dna'].data['patient_id'] = valid_nodes_dna[:, -1]
    valid_graph.nodes['rna'].data['patient_id'] = valid_nodes_rna[:, -1]
    valid_graph.nodes['protein'].data['patient_id'] = valid_nodes_protein[:, -1]

    model = clf_graph(train_graph)  # 模型使用
    configs = load_configuratiton(model, device)
    model = model.to(device)

    optimizer: device | str | Adam | StepLR | int | float = configs['optimizer']
    scheduler = configs['scheduler']
    device = configs['device']

    best_r2 = -float("inf")
    early_stop_counter = 0
    
    batch_size = 200
    num_samples = train_nodes_dna.shape[1]
    patient_indices = list(range(num_samples))

    best_valid_score = -float("inf")
    best_epoch = 0
    best_state_dict = None
    for epoch in range(400):
        model.train()
        total_loss = 0.0
        for i in range(0, num_samples, batch_size):
            batch_idx = patient_indices[i:i + batch_size]
            idx = torch.tensor(batch_idx).to(device)

            optimizer.zero_grad()
            with autocast():
                output_dna, output_rna, output_pro, train_edge_weight, train_logits_dna, train_logits_rna, train_logits_pro = model(train_graph)
                
                # check for NaN
                if check_for_nan(output_dna, "Output DNA", epoch=epoch, batch_idx=i) or check_for_nan(output_rna, "Output RNA", epoch=epoch, batch_idx=i) or check_for_nan(output_pro, "Output Protein", epoch=epoch, batch_idx=i):
                    continue

                if idx.max().item() >= output_dna.shape[0]:
                    print(f"[wrong] idx is Out of range! idx.max={idx.max().item()}, output_dna.shape={output_dna.shape}")
                    exit(1)

                loss_dna = criterion(output_dna[idx], train_imdna[idx])
                loss_rna = criterion(output_rna[idx], train_imrna[idx])
                loss_pro = criterion(output_pro[idx], train_impro[idx])
                loss = loss_dna + loss_rna + loss_pro

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            torch.cuda.empty_cache()

        model.eval()
        with torch.no_grad():
            train_output_dna, train_output_rna, train_output_pro = batched_predict(model, train_graph, batch_size, device)
            
            train_graph.nodes['rna'].data['logits'] = train_logits_rna.detach()
            train_graph.nodes['protein'].data['logits'] = train_logits_pro.detach()
            train_graph.nodes['dna'].data['logits'] = train_logits_dna.detach()

            train_rna_mae, train_rna_rmse, train_rna_r2 = calculate_metrics(train_output_rna, train_imrna)
            train_dna_mae, train_dna_rmse, train_dna_r2 = calculate_metrics(train_output_dna, train_imdna)
            train_pro_mae, train_pro_rmse, train_pro_r2 = calculate_metrics(train_output_pro, train_impro)

            with open('training_log_304_100.txt', 'a') as log_file:
                log_file.write(f"Epoch: {epoch}, Train rna r2={train_rna_r2}, Train dna r2={train_dna_r2}, Train pro r2={train_pro_r2}, train_loss={total_loss:.4f}\n")
            print("Epoch:", epoch, "Train rna r2=", train_rna_r2, "Train dna r2=", train_dna_r2, "Train pro r2=",
                  train_pro_r2, "train_loss", total_loss)

            valid_output_dna, valid_output_rna, valid_output_pro, valid_edge_weight, valid_logits_dna, valid_logits_rna, valid_logits_pro = model(valid_graph)
            
            valid_graph.nodes['rna'].data['logits'] = valid_logits_rna.detach()
            valid_graph.nodes['protein'].data['logits'] = valid_logits_pro.detach()
            valid_graph.nodes['dna'].data['logits'] = valid_logits_dna.detach()


            # valid_loss = criterion(valid_output,valid_status)
            rna_weight = 1.5
            valid_rna_loss = criterion(valid_output_rna, valid_imrna)
            valid_dna_loss = criterion(valid_output_dna, valid_imdna)
            valid_pro_loss = criterion(valid_output_pro, valid_impro)

            valid_rna_losses.append(valid_rna_loss.item())
            #valid_rna_r2 = calculate_metrics(valid_output_rna, valid_imrna)[2]

            valid_loss = rna_weight * valid_rna_loss + valid_dna_loss + valid_pro_loss

            valid_rna_mae, valid_rna_rmse, valid_rna_r2 = calculate_metrics(valid_output_rna, valid_imrna)
            valid_dna_mae, valid_dna_rmse, valid_dna_r2 = calculate_metrics(valid_output_dna, valid_imdna)
            valid_pro_mae, valid_pro_rmse, valid_pro_r2 = calculate_metrics(valid_output_pro, valid_impro)

            torch.cuda.empty_cache()

            valid_r2 = min(valid_rna_r2, valid_dna_r2, valid_pro_r2)
            current_valid_score = min(valid_rna_r2, valid_dna_r2, valid_pro_r2)
            if current_valid_score > best_valid_score:
                best_valid_score = current_valid_score
                best_epoch = epoch
                best_state_dict = copy.deepcopy(model.state_dict())
                print(f"Epoch {epoch} for the beat, R² = {best_valid_score}")
            with open('training_log_304_100.txt', 'a') as log_file:
                log_file.write(f"Epoch: {epoch}, Valid rna r2={valid_rna_r2}, Valid dna r2={valid_dna_r2}, Valid pro r2={valid_pro_r2}, valid_loss={valid_loss.cpu()}\n")
            print("Epoch:", epoch, "Valid rna r2=", valid_rna_r2, "Valid dna r2=", valid_dna_r2, "Valid pro r2=",
                  valid_pro_r2, "valid_loss", valid_loss.cpu())

            # if (valid_ci>0.6 and train_ci>0.7) or (valid_ci<0.4 and train_ci<0.3):

            #torch.save(model, "/home/dengjingran/Multi_omic_Immunity/GCN_immune/"+outputfolder+"goodepoch_" + str(epoch) + fold + seed+'.pth')

            # valid_fe_rna = pd.DataFrame(valid_logits_rna.detach().cpu().numpy())
            # valid_fe_rna.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ 'valid_fe_rna.csv')
            # valid_fe_pro = pd.DataFrame(valid_logits_pro.detach().cpu().numpy())
            # valid_fe_pro.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ 'valid_fe_pro.csv')
            # valid_fe_phos = pd.DataFrame(valid_logits_phos.detach().cpu().numpy())
            # valid_fe_phos.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ 'valid_fe_phos.csv')

            # train_fe_rna = pd.DataFrame(train_logits_rna.detach().cpu().numpy())
            # train_fe_rna.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ 'train_fe_rna.csv')
            # train_fe_pro = pd.DataFrame(train_logits_pro.detach().cpu().numpy())
            # train_fe_pro.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ 'train_fe_pro.csv')
            # train_fe_phos = pd.DataFrame(train_logits_phos.detach().cpu().numpy())
            # train_fe_phos.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ 'train_fe_phos.csv')

            # df_train_os = pd.DataFrame({'train_signature': train_output[:,0].detach().cpu().numpy(),'train_os':train_os.detach().cpu().numpy(),'train_status':train_status.detach().cpu().numpy()})
            # df_valid_os = pd.DataFrame({'valid_signature': valid_output[:,0].detach().cpu().numpy(),'valid_os':valid_os.detach().cpu().numpy(),'valid_status':valid_status.detach().cpu().numpy()})

            # for key,value in train_edge_weight.items():
            #     train_edge={}
            #     value = value.detach().cpu().numpy()
            #     train_edge = pd.DataFrame(value)

            #     train_edge.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ key + 'edge_weight.csv')

            # df_train_os.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ '_trainci.csv',index=False)

            # df_valid_os.to_csv("F:/GCN_multiomic/"+outputfolder+"okepoch" + str(epoch) + fold + seed+ '_validci.csv',index=False)
            #if valid_ci<0.4 and train_ci<0.3:
            #   valid_ci=1-valid_ci
            #     train_ci=1-train_ci
            # if valid_ci>best_valid:
            #     best_train=train_ci
            #     best_valid=valid_ci
            #     best_epoch=epoch
            #with open('training_log.txt', 'a') as log_file:
                #log_file.write(f"Loss difference: {loss_before - train_loss.detach().cpu()}\n")
            print(loss_before - total_loss)
            # save the best model
            best_train=min(train_rna_r2, train_dna_r2, train_pro_r2)
            best_valid=min(valid_rna_r2, valid_dna_r2, valid_pro_r2)
            best_epoch = epoch
            if epoch >= initial_epochs:

                if valid_rna_r2 > valid_rna_before and valid_dna_r2 > valid_dna_before and valid_pro_r2 > valid_pro_before:
                    valid_rna_before = valid_rna_r2  # Keep the previous validation
                    valid_dna_before = valid_dna_r2
                    valid_pro_before = valid_pro_r2
                    #best_valid = valid_rna_r2
                    #best_valid = valid_r2
                    best_epoch = epoch
                    no_improvement_counter = 0
                else:
                    valid_rna_before = valid_rna_r2  # Keep the previous validation
                    valid_dna_before = valid_dna_r2
                    valid_pro_before = valid_pro_r2
                    best_epoch = epoch
                    no_improvement_counter += 1

                #print(f" no_improvement_counter is {no_improvement_counter},valid_rna_before is {valid_rna_before},valid_rna_r2 is {valid_rna_r2}")

                if no_improvement_counter >= patience:

                    train_data = pd.DataFrame({
                        'train_output_rna_1': train_output_rna.cpu().detach().numpy()[:, 0],  # Get the predicted values from the first colum
                        'train_output_rna_2': train_output_rna.cpu().detach().numpy()[:, 1],  # Get the predicted values from the second column.
                        'train_output_dna_1': train_output_dna.cpu().detach().numpy()[:, 0],
                        'train_output_dna_2': train_output_dna.cpu().detach().numpy()[:, 1],
                        'train_output_pro_1': train_output_pro.cpu().detach().numpy()[:, 0],
                        'train_output_pro_2': train_output_pro.cpu().detach().numpy()[:, 1],
                        'train_imrna_1': train_imrna.cpu().detach().numpy()[:, 0],  # Get the labels from the first column
                        'train_imrna_2': train_imrna.cpu().detach().numpy()[:, 1],  # Get the labels from the second column
                        'train_imdna_1': train_imdna.cpu().detach().numpy()[:, 0],
                        'train_imdna_2': train_imdna.cpu().detach().numpy()[:, 1],
                        'train_impro_1': train_impro.cpu().detach().numpy()[:, 0],
                        'train_impro_2': train_impro.cpu().detach().numpy()[:, 1]
                    })

                    valid_data = pd.DataFrame({
                        'valid_output_rna_1': valid_output_rna.cpu().detach().numpy()[:, 0],  # Get the predicted values from the first column
                        'valid_output_rna_2': valid_output_rna.cpu().detach().numpy()[:, 1],  # Get the predicted values from the second column
                        'valid_output_dna_1': valid_output_dna.cpu().detach().numpy()[:, 0],
                        'valid_output_dna_2': valid_output_dna.cpu().detach().numpy()[:, 1],
                        'valid_output_pro_1': valid_output_pro.cpu().detach().numpy()[:, 0],
                        'valid_output_pro_2': valid_output_pro.cpu().detach().numpy()[:, 1],
                        'valid_imrna_1': valid_imrna.cpu().detach().numpy()[:, 0],  # Get the labels from the first column
                        'valid_imrna_2': valid_imrna.cpu().detach().numpy()[:, 1],  # Get the labels from the second column
                        'valid_imdna_1': valid_imdna.cpu().detach().numpy()[:, 0],
                        'valid_imdna_2': valid_imdna.cpu().detach().numpy()[:, 1],
                        'valid_impro_1': valid_impro.cpu().detach().numpy()[:, 0],
                        'valid_impro_2': valid_impro.cpu().detach().numpy()[:, 1]
                    })
                    # save to CSV
                    train_data.to_csv(f"/home/dengjingran/Multi_omic_Immunity/GCN_immune/{outputfolder}train_0/okepoch{str(epoch)}{fold}{seed}_train_combined.csv",index=False)
                    valid_data.to_csv(f"/home/dengjingran/Multi_omic_Immunity/GCN_immune/{outputfolder}valid_0/okepoch{str(epoch)}{fold}{seed}_valid_combined.csv",index=False)
                    
                    '''valid_out_rna = pd.DataFrame(valid_output_rna.detach().cpu().numpy())
                    valid_out_rna.to_csv("/home/dengjingran/Multi_omic_Immunity/GCN_immune/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'valid_output_rna.csv')

                    valid_out_pro = pd.DataFrame(valid_output_pro.detach().cpu().numpy())
                    valid_out_pro.to_csv("/home/dengjingran/Multi_omic_Immunity/GCN_immune/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'valid_output_pro.csv')

                    valid_out_dna = pd.DataFrame(valid_output_dna.detach().cpu().numpy())
                    valid_out_dna.to_csv("/home/dengjingran/Multi_omic_Immunity/GCN_immune/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'valid_output_dna.csv')

                    # 新增：保存 train 的最终输出（假设 train_output_* 已获取）
                    train_out_rna = pd.DataFrame(train_output_rna.detach().cpu().numpy())
                    train_out_rna.to_csv("/home/dengjingran/Multi_omic_Immunity/GCN_immune/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'train_output_rna.csv')

                    train_out_pro = pd.DataFrame(train_output_pro.detach().cpu().numpy())
                    train_out_pro.to_csv("/home/dengjingran/Multi_omic_Immunity/GCN_immune/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'train_output_pro.csv')

                    train_out_dna = pd.DataFrame(train_output_dna.detach().cpu().numpy())
                    train_out_dna.to_csv("/home/dengjingran/Multi_omic_Immunity/GCN_immune/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'train_output_dna.csv')'''
                    
                    train_fe_rna = pd.DataFrame(train_logits_rna.detach().cpu().numpy())
                    train_fe_rna.to_csv("/ImmGraph/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'train_fe_rna.csv')

                    train_fe_pro = pd.DataFrame(train_logits_pro.detach().cpu().numpy())
                    train_fe_pro.to_csv("/ImmGraph/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'train_fe_pro.csv')

                    train_fe_dna = pd.DataFrame(train_logits_dna.detach().cpu().numpy())
                    train_fe_dna.to_csv("/ImmGraph/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'train_fe_dna.csv')


                    valid_fe_rna = pd.DataFrame(valid_logits_rna.detach().cpu().numpy())
                    valid_fe_rna.to_csv("/ImmGraph/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'valid_fe_rna.csv')
                    valid_fe_pro = pd.DataFrame(valid_logits_pro.detach().cpu().numpy())
                    valid_fe_pro.to_csv("/ImmGraph/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'valid_fe_pro.csv')
                    valid_fe_dna = pd.DataFrame(valid_logits_dna.detach().cpu().numpy())
                    valid_fe_dna.to_csv("/ImmGraph/" + outputfolder + "node_feature/" + "okepoch" + str(epoch) + fold + seed + 'valid_fe_dna.csv')

                    train_preds = pd.DataFrame({
                        'patient_id': list(im.iloc[train_list, 0]),
                        'rna_pred_1': train_output_rna[:, 0].cpu().numpy(),
                        'rna_pred_2': train_output_rna[:, 1].cpu().numpy(),
                        'dna_pred_1': train_output_dna[:, 0].cpu().numpy(),
                        'dna_pred_2': train_output_dna[:, 1].cpu().numpy(),
                        'pro_pred_1': train_output_pro[:, 0].cpu().numpy(),
                        'pro_pred_2': train_output_pro[:, 1].cpu().numpy(),
                    })
                    train_preds.to_csv(f"/ImmGraph/{outputfolder}patient_pred/okepoch{str(epoch)}{fold}{seed}_train_patient_preds.csv", index=False)


                    valid_preds = pd.DataFrame({
                        'patient_id': list(im.iloc[valid_list, 0]),
                        'rna_pred_1': valid_output_rna[:, 0].cpu().numpy(),
                        'rna_pred_2': valid_output_rna[:, 1].cpu().numpy(),
                        'dna_pred_1': valid_output_dna[:, 0].cpu().numpy(),
                        'dna_pred_2': valid_output_dna[:, 1].cpu().numpy(),
                        'pro_pred_1': valid_output_pro[:, 0].cpu().numpy(),
                        'pro_pred_2': valid_output_pro[:, 1].cpu().numpy(),
                    })
                    valid_preds.to_csv(f"/ImmGraph/{outputfolder}patient_pred/okepoch{str(epoch)}{fold}{seed}_valid_patient_preds.csv", index=False)


                    for key, value in train_edge_weight.items():

                        edge_weight = value.detach().cpu().numpy()


                        edge_weight = edge_weight.squeeze()

                    for key, value in valid_edge_weight.items():

                        edge_weight = value.detach().cpu().numpy()

                        edge_weight = edge_weight.squeeze()

                    print(f"At epoch {epoch} , the validation R² = {best_valid}. No improvement for 10 consecutive times, stopping training early.")


                    for etype in train_edge_weight:
                        weight_tensor = train_edge_weight[etype].squeeze(0).detach().to(device)
                        train_graph.edges[etype].data['weight'] = weight_tensor

                    for etype in valid_edge_weight:
                        weight_tensor = valid_edge_weight[etype].squeeze(0).detach().to(device)
                        valid_graph.edges[etype].data['weight'] = weight_tensor

                    for etype in train_graph.canonical_etypes:
                        keys = list(train_graph.edges[etype].data.keys())
                        for key in keys:
                            if key != 'weight':
                                del train_graph.edges[etype].data[key]

                    for etype in valid_graph.canonical_etypes:
                        keys = list(valid_graph.edges[etype].data.keys())
                        for key in keys:
                            if key != 'weight':
                                del valid_graph.edges[etype].data[key]
                                
                    # 保存异构图到文件

                    file_path_1 = f"data/ImmGraph_results/heterographs_ImmGraph/train_graph_epoch_{epoch}{fold}.dgl"
                    file_path_2 = f"data/ImmGraph_results/heterographs_ImmGraph/valid_graph_epoch_{epoch}{fold}.dgl"
                    dgl.save_graphs(file_path_1, [train_graph])
                    dgl.save_graphs(file_path_2, [valid_graph])

                    print(f"Saved graphs for epoch {epoch}")
                    break
    if best_state_dict is not None:
        save_path = f"/home/dengjingran/Multi_omic_Immunity/GCN_immune/{outputfolder}best_model_{fold}_{seed}.pth"
        torch.save(best_state_dict, save_path)
        print(f"[{fold}] Training completed: The best model for this fold (epoch= {best_epoch}, validation R²= {best_valid_score} has been saved to {save_path}")
    

    if best_state_dict is not None:
        model_cpu = (model.module if hasattr(model, "module") else model)
        model_cpu.to("cpu")
        model_cpu.load_state_dict(best_state_dict)
        model_cpu.to(device)


    model.eval()
    with torch.inference_mode():
        valid_output_dna, valid_output_rna, valid_output_pro, *_ = model(valid_graph)
        
        train_output_dna, train_output_rna, train_output_pro, *_ = model(train_graph)


# prediction of validation
    valid_pred_rna = valid_output_rna.detach().float().cpu().reshape(-1)
    valid_pred_dna = valid_output_dna.detach().float().cpu().reshape(-1)
    valid_pred_pro = valid_output_pro.detach().float().cpu().reshape(-1)

    valid_tgt_rna = valid_imrna.detach().float().cpu().reshape(-1)
    valid_tgt_dna = valid_imdna.detach().float().cpu().reshape(-1)
    valid_tgt_pro = valid_impro.detach().float().cpu().reshape(-1)

# prediction of training
    train_pred_rna = train_output_rna.detach().float().cpu().reshape(-1)
    train_pred_dna = train_output_dna.detach().float().cpu().reshape(-1)
    train_pred_pro = train_output_pro.detach().float().cpu().reshape(-1)

    train_tgt_rna = train_imrna.detach().float().cpu().reshape(-1)
    train_tgt_dna = train_imdna.detach().float().cpu().reshape(-1)
    train_tgt_pro = train_impro.detach().float().cpu().reshape(-1)
  
    del valid_output_rna, valid_output_dna, valid_output_pro
    del train_output_rna, train_output_dna, train_output_pro  # 【新增】
    torch.cuda.empty_cache()

# metrics of prediction dastaset
    valid_rna_m = compute_all_metrics(valid_pred_rna, valid_tgt_rna)
    valid_dna_m = compute_all_metrics(valid_pred_dna, valid_tgt_dna)
    valid_pro_m = compute_all_metrics(valid_pred_pro, valid_tgt_pro)

# metrics of training dataset
    train_rna_m = compute_all_metrics(train_pred_rna, train_tgt_rna)
    train_dna_m = compute_all_metrics(train_pred_dna, train_tgt_dna)
    train_pro_m = compute_all_metrics(train_pred_pro, train_tgt_pro)
  
    row = {
    
        "seed": seed, "fold": str(fold),
        "valid_rna_MedAE": valid_rna_m["MedAE"], "valid_rna_Pearson": valid_rna_m["Pearson"],
        "valid_rna_MAE": valid_rna_m["MAE"], "valid_rna_RMSE": valid_rna_m["RMSE"], "valid_rna_R2": valid_rna_m["R2"],
        "valid_dna_MedAE": valid_dna_m["MedAE"], "valid_dna_Pearson": valid_dna_m["Pearson"],
        "valid_dna_MAE": valid_dna_m["MAE"], "valid_dna_RMSE": valid_dna_m["RMSE"], "valid_dna_R2": valid_dna_m["R2"],
        "valid_pro_MedAE": valid_pro_m["MedAE"], "valid_pro_Pearson": valid_pro_m["Pearson"],
        "valid_pro_MAE": valid_pro_m["MAE"], "valid_pro_RMSE": valid_pro_m["RMSE"], "valid_pro_R2": valid_pro_m["R2"],
    

        "train_rna_MedAE": train_rna_m["MedAE"], "train_rna_Pearson": train_rna_m["Pearson"],
        "train_rna_MAE": train_rna_m["MAE"], "train_rna_RMSE": train_rna_m["RMSE"], "train_rna_R2": train_rna_m["R2"],
        "train_dna_MedAE": train_dna_m["MedAE"], "train_dna_Pearson": train_dna_m["Pearson"],
        "train_dna_MAE": train_dna_m["MAE"], "train_dna_RMSE": train_dna_m["RMSE"], "train_dna_R2": train_dna_m["R2"],
        "train_pro_MedAE": train_pro_m["MedAE"], "train_pro_Pearson": train_pro_m["Pearson"],
        "train_pro_MAE": train_pro_m["MAE"], "train_pro_RMSE": train_pro_m["RMSE"], "train_pro_R2": train_pro_m["R2"],
        
        "best_min_R2": best_valid_score, "best_epoch": best_epoch,
    }

    metrics_csv = f"/ImmGraph/{outputfolder}fold_metrics_with_train.csv"
    os.makedirs(os.path.dirname(metrics_csv), exist_ok=True)
    write_header = not os.path.exists(metrics_csv)

    with open(metrics_csv, "a", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


    del train_pred_rna, train_pred_dna, train_pred_pro
    del train_tgt_rna, train_tgt_dna, train_tgt_pro
    del valid_pred_rna, valid_pred_dna, valid_pred_pro
    del valid_tgt_rna, valid_tgt_dna, valid_tgt_pro



    return best_train, best_valid, best_epoch, valid_rna_losses

    # scheduler.step()


def main(device):
    batch_size = 8
    all_rna_losses = []
    DATA_PATH = "/ImmGraph/" + inputfolder

    nodes_dna = pd.read_csv(DATA_PATH + "nodes_dna.csv", sep=',', header=0)
    nodes_rna = pd.read_csv(DATA_PATH + "nodes_rna.csv", sep=',', header=0)
    nodes_protein = pd.read_csv(DATA_PATH + "nodes_protein.csv", sep=',', header=0)

    edges_dna = pd.read_csv(DATA_PATH + "edges_dna.csv", sep=',', header=0)
    edges_rna = pd.read_csv(DATA_PATH + "edges_rna.csv", sep=',', header=0)
    edges_protein = pd.read_csv(DATA_PATH + "edges_protein.csv", sep=',', header=0)

    edges_dnarna = pd.read_csv(DATA_PATH + "edges_dnarna.csv", sep=',', header=0)
    edges_rnapro = pd.read_csv(DATA_PATH + "edges_rnapro.csv", sep=',', header=0)

    im = pd.read_csv(DATA_PATH + "im.csv", sep=',', header=0)
    patient_ids = im.iloc[:, 0].values

    all_list = list(range(im.shape[0]))
    # all_list.pop(0)
    fold1train = []
    fold1valid = []
    fold2train = []
    fold2valid = []
    fold3train = []
    fold3valid = []
    fold4train = []
    fold4valid = []
    fold5train = []
    fold5valid = []
    epoche1 = []
    epoche2 = []
    epoche3 = []
    epoche4 = []
    epoche5 = []
    seed = []
    ################################curseed################################
    for cur_seed in range(0,1):
        setup_seed(cur_seed)
        fold1 = random.sample(all_list, 61)
        rest1 = [item for item in all_list if item not in fold1]
        fold2 = random.sample(rest1, 61)
        rest2 = [item for item in rest1 if item not in fold2]
        fold3 = random.sample(rest2, 61)
        rest3 = [item for item in rest2 if item not in fold3]
        fold4 = random.sample(rest3, 61)
        fold5 = [item for item in rest3 if item not in fold4]

        all_list = list(range(im.shape[0]))
        fold_data = []

        TRAIN_LIST1 = fold1 + fold2 + fold3 + fold4
        VALID_LIST1 = fold5

        TRAIN_LIST2 = fold1 + fold2 + fold3 + fold5
        VALID_LIST2 = fold4

        TRAIN_LIST3 = fold1 + fold2 + fold4 + fold5
        VALID_LIST3 = fold3

        TRAIN_LIST4 = fold1 + fold3 + fold4 + fold5
        VALID_LIST4 = fold2

        TRAIN_LIST5 = fold2 + fold3 + fold4 + fold5
        VALID_LIST5 = fold1
        
        
        fold1_ids = im.iloc[VALID_LIST1, 0].values
        fold2_ids = im.iloc[VALID_LIST2, 0].values
        fold3_ids = im.iloc[VALID_LIST3, 0].values
        fold4_ids = im.iloc[VALID_LIST4, 0].values
        fold5_ids = im.iloc[VALID_LIST5, 0].values

        # 将每个fold的数据保存到列表
        fold_data.append({
            'Fold': 'Fold1',
            'Train': TRAIN_LIST1,
            'Valid': VALID_LIST1
        })
        fold_data.append({
            'Fold': 'Fold2',
            'Train': TRAIN_LIST2,
            'Valid': VALID_LIST2
        })
        fold_data.append({
            'Fold': 'Fold3',
            'Train': TRAIN_LIST3,
            'Valid': VALID_LIST3
        })
        fold_data.append({
            'Fold': 'Fold4',
            'Train': TRAIN_LIST4,
            'Valid': VALID_LIST4
        })
        fold_data.append({
            'Fold': 'Fold5',
            'Train': TRAIN_LIST5,
            'Valid': VALID_LIST5
        })

        print(cur_seed)

        train_loader1 = train_datatotensor(TRAIN_LIST1, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        valid_loader1 = train_datatotensor(VALID_LIST1, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        best_train1, best_valid1, best_epoch1, rna_losses1 = train(train_loader1, valid_loader1, fold="f1s",
                                                                   seed=str(cur_seed), device=device, im=im, train_list=TRAIN_LIST1, valid_list=VALID_LIST1)
        all_rna_losses.append(rna_losses1)

        with open('training_log_304_100.txt', 'a') as log_file:
            log_file.write(
                f"Fold1___________________________________________________________Best Train r2={best_train1}, "
                f"Valid r2={best_valid1}, Epoch={best_epoch1}, seed={cur_seed}\n")


        print("Fold1___________________________________________________________Best Train r2=", best_train1,
              "Valid r2=", best_valid1, "Epoch=", best_epoch1, "seed=", cur_seed)

        train_loader2 = train_datatotensor(TRAIN_LIST2, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        valid_loader2 = train_datatotensor(VALID_LIST2, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        best_train2, best_valid2, best_epoch2, rna_losses2 = train(train_loader2, valid_loader2, fold="f2s",
                                                                   seed=str(cur_seed), device=device, im=im, train_list=TRAIN_LIST2, valid_list=VALID_LIST2)
        all_rna_losses.append(rna_losses2) 
        with open('training_log_304_100.txt', 'a') as log_file:
            log_file.write(
                f"Fold2___________________________________________________________Best Train r2={best_train1}, "
                f"Valid r2={best_valid1}, Epoch={best_epoch1}, seed={cur_seed}\n")
        print("Fold2__________________________________________________________Best Train r2=", best_train2, "Valid r2=",
              best_valid2, "Epoch=", best_epoch2, "seed=", cur_seed)

        train_loader3 = train_datatotensor(TRAIN_LIST3, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        valid_loader3 = train_datatotensor(VALID_LIST3, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        best_train3, best_valid3, best_epoch3, rna_losses3 = train(train_loader3, valid_loader3, fold="f3s",
                                                                   seed=str(cur_seed), device=device, im=im, train_list=TRAIN_LIST3, valid_list=VALID_LIST3)
        all_rna_losses.append(rna_losses3)
        with open('training_log_304_100.txt', 'a') as log_file:
            log_file.write(
                f"Fold3___________________________________________________________Best Train r2={best_train1}, "
                f"Valid r2={best_valid1}, Epoch={best_epoch1}, seed={cur_seed}\n")
        print("Fold3_________________________________________________________Best Train r2=", best_train3, "Valid r2=",
              best_valid3, "Epoch=", best_epoch3, "seed=", cur_seed)

        train_loader4 = train_datatotensor(TRAIN_LIST4, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        valid_loader4 = train_datatotensor(VALID_LIST4, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        best_train4, best_valid4, best_epoch4, rna_losses4 = train(train_loader4, valid_loader4, fold="f4s",
                                                                   seed=str(cur_seed), device=device, im=im, train_list=TRAIN_LIST4, valid_list=VALID_LIST4)
        all_rna_losses.append(rna_losses4)
        with open('training_log_304_100.txt', 'a') as log_file:
            log_file.write(
                f"Fold4___________________________________________________________Best Train r2={best_train4}, "
                f"Valid r2={best_valid4}, Epoch={best_epoch4}, seed={cur_seed}\n")
        print("Fold4_________________________________________________________Best Train r2=", best_train4, "Valid r2=",
              best_valid4, "Epoch=", best_epoch4, "seed=", cur_seed)

        train_loader5 = train_datatotensor(TRAIN_LIST5, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        valid_loader5 = train_datatotensor(VALID_LIST5, nodes_dna, nodes_rna, nodes_protein,
                                           edges_dna, edges_rna, edges_protein, edges_dnarna, edges_rnapro, im)

        best_train5, best_valid5, best_epoch5, rna_losses5 = train(train_loader5, valid_loader5, fold="f5s",
                                                                   seed=str(cur_seed), device=device, im=im, train_list=TRAIN_LIST5, valid_list=VALID_LIST5)
        all_rna_losses.append(rna_losses5)
        with open('training_log_304_100.txt', 'a') as log_file:
            log_file.write(
                f"Fold5___________________________________________________________Best Train r2={best_train5}, "
                f"Valid r2={best_valid5}, Epoch={best_epoch5}, seed={cur_seed}\n")
        print("Fold5_________________________________________________________Best Train r2=", best_train5, "Valid r2=",
              best_valid5, "Epoch=", best_epoch5, "seed=", cur_seed)
        
        fold1train.append(best_train1)
        fold1valid.append(best_valid1)
        fold2train.append(best_train2)
        fold2valid.append(best_valid2)
        fold3train.append(best_train3)
        fold3valid.append(best_valid3)
        fold4train.append(best_train4)
        fold4valid.append(best_valid4)
        fold5train.append(best_train5)
        fold5valid.append(best_valid5)
        epoche1.append(best_epoch1)
        epoche2.append(best_epoch2)
        epoche3.append(best_epoch3)
        epoche4.append(best_epoch4)
        epoche5.append(best_epoch5)
        seed.append(cur_seed)
    Results = pd.DataFrame()
    Results["seed"] = seed
    Results["epoche1"] = epoche1
    Results["fold1train"] = fold1train
    Results["fold1valid"] = fold1valid
    Results["epoche2"] = epoche2
    Results["fold2train"] = fold2train
    Results["fold2valid"] = fold2valid
    Results["epoche3"] = epoche3
    Results["fold3train"] = fold3train
    Results["fold3valid"] = fold3valid
    Results["epoche4"] = epoche4
    Results["fold4train"] = fold4train
    Results["fold4valid"] = fold4valid
    Results["epoche5"] = epoche5
    Results["fold5train"] = fold5train
    Results["fold5valid"] = fold5valid

    # 将所有fold的数据保存为DataFrame
    fold_df = pd.DataFrame(fold_data)



if __name__ == "__main__":
    device = torch.device('cuda')
    print(torch.cuda.is_available())
    main(device)
