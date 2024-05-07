from .model import CSPred
import numpy as np
import random
import copy
import os
from operator import itemgetter
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_recall_curve, auc, matthews_corrcoef

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

def fewshot(activesite, activesite_not_in_aloop, 
            kinases_cptac, kinases_all,
            not_in_aloop_weight, t5_emb, sample_emb, 
            nfold = 5, min_delta = 0, patience = 30, batch_size = 64, 
            lr = 1e-04, weight_decay = 0.01, dropout_rate = 0.1,
            path = './', device = 'cuda:0'):
    """
    :param activesite: 177 known critical phos-sites
    :param activesite_not_in_aloop: 84 known critical phos-sites that are not in A-loop
    :param kinases_cptac: kinase phos-sites detected in CPTAC
    :param kinases_all: all known critical phos-sites and kinase phos-sites detected in CPTAC
    :param not_in_aloop_weight: loss weight for phos-sites that are not in A-loop
    :param t5_emb: ProtT5 embeddings
    :param sample_emb: PhoSiteformer embeddings
    :param path: path to save results
    :param device: cpu or cuda
    """
    funcinf=set(activesite)
    pos = []
    neg = []
    for site in kinases_all:
        if site in funcinf:
            pos.append(site)
        else:
            neg.append(site)

    print(len(pos))
    
    fw = open(f"{path}/MAML_AUCs.txt", "a")
    pos_size = len(pos)
        
    for a in range(10000):
        if len(neg) > pos_size*11:
            new_neg = random.sample(neg, pos_size*10)
            tem_neg = copy.deepcopy(neg)
            for j in range(len(tem_neg)):
                negpep = tem_neg[j]
                if negpep in new_neg:
                    neg.remove(negpep)
            print(len(neg))
        
            x1 = torch.tensor(itemgetter(*(pos + new_neg))(t5_emb))
            x2 = torch.tensor(itemgetter(*(pos + new_neg))(sample_emb))
            
            X = torch.cat((x1, x2), dim=1)
            Y = np.array([1] * len(pos) + [0] * len(new_neg))
            both_emb = np.array([1 if i in kinases_cptac else 0 for i in pos + new_neg])
            site_weight = np.array([not_in_aloop_weight if i in activesite_not_in_aloop else 1 for i in pos + new_neg])

            x1_train, x1_test, x2_train, x2_test, \
            both_emb_train, both_emb_test, site_weight_train, site_weight_test, \
            X_train, X_test, Y_train, Y_test = \
            train_test_split(x1, x2, both_emb, site_weight, X, Y, test_size=0.2, random_state=42)
            
            auc_all, accuracy_all, f1_all, auprc_all, mcc_all, best_model = \
                    ft(x1_train, x1_test, x2_train, x2_test, 
                       both_emb_train, both_emb_test, 
                       site_weight_train, site_weight_test,
                       X_train, X_test, Y_train, Y_test, 
                       a, nfold = nfold, min_delta = min_delta, 
                       patience = patience, batch_size = batch_size, 
                       lr = lr, weight_decay = weight_decay, 
                       dropout_rate = dropout_rate,
                       path = path, device = device)
            
            fw.write(str(a + 1) + "\tBest:" + "\t" + \
                     str(auc_all) + "\t" + \
                     str(accuracy_all) + "\t" + \
                     str(f1_all)+ "\t" + \
                     str(auprc_all)+ "\t" + \
                     str(mcc_all)+ "\t" + \
                     str(best_model) + "\n")
            fw.flush()
        
        else:
            x1 = torch.tensor(itemgetter(*(pos + neg))(t5_emb))
            x2 = torch.tensor(itemgetter(*(pos + neg))(sample_emb))
            
            X = torch.cat((x1, x2), dim=1)
            Y = np.array([1] * len(pos) + [0] * len(neg))
            both_emb = np.array([1 if i in kinases_cptac else 0 for i in pos + neg])
            site_weight = np.array([2 if i in activesite_not_in_aloop else 1 for i in pos + neg])

            x1_train, x1_test, x2_train, x2_test, \
            both_emb_train, both_emb_test, site_weight_train, site_weight_test, \
            X_train, X_test, Y_train, Y_test = \
            train_test_split(x1, x2, both_emb, site_weight, X, Y, test_size=0.2, random_state=42)
            
            auc_all, accuracy_all, f1_all, auprc_all, mcc_all, best_model = \
                    ft(x1_train, x1_test, x2_train, x2_test, 
                       both_emb_train, both_emb_test, 
                       site_weight_train, site_weight_test,
                       X_train, X_test, Y_train, Y_test, 
                       a, nfold = nfold, min_delta = min_delta, 
                       patience = patience, batch_size = batch_size, 
                       lr = lr, weight_decay = weight_decay, 
                       dropout_rate = dropout_rate,
                       path = path, device = device)
            
            fw.write(str(a + 1) + "\tBest:" + "\t" + \
                     str(auc_all) + "\t" + \
                     str(accuracy_all) + "\t" + \
                     str(f1_all)+ "\t" + \
                     str(auprc_all)+ "\t" + \
                     str(mcc_all)+ "\t" + \
                     str(best_model) + "\n")
            fw.flush()
            
            break
    
    fw.flush()
    fw.close()
            
          
def ft(x1_train, x1_test, x2_train, x2_test, 
       both_emb_train, both_emb_test, 
       site_weight_train, site_weight_test,
       X_train, X_test, Y_train, Y_test, 
       a, nfold = 5, min_delta = 0, patience = 30, batch_size = 64, 
       lr = 1e-04, weight_decay = 0.01, dropout_rate = 0.1,
       path = './', device = 'cuda:0'):
    
    skf = StratifiedKFold(n_splits = nfold)
    num = 0 ## number of k-fold iterations
    best_auc = 0.0 ## best auc of all k-fold iterations
    best_model = 0 ## best model number
    
    auc_all = 0.0
    accuracy_all = 0.0
    f1_all = 0.0
    auprc_all = 0.0
    mcc_all = 0.0
    for train_index, test_index in skf.split(X_train, Y_train):
        num += 1
        print("kfold_" + str(num))
        ## 1. training set and validation set
        X_tra, X_val = X_train[train_index], X_train[test_index]
        x1_tra, x1_val = x1_train[train_index], x1_train[test_index]
        x2_tra, x2_val = x2_train[train_index], x2_train[test_index]
        Y_tra, Y_val = Y_train[train_index], Y_train[test_index]
        both_emb_tra, both_emb_val = both_emb_train[train_index], both_emb_train[test_index]
        site_weight_tra, site_weight_val = site_weight_train[train_index], site_weight_train[test_index]

        ## 2. data loader
        train_dataset = TensorDataset(torch.Tensor(X_tra).to(device), 
                                      torch.Tensor(x1_tra).to(device), 
                                      torch.Tensor(x2_tra).to(device), 
                                      torch.Tensor(both_emb_tra).to(device), 
                                      torch.Tensor(site_weight_tra).to(device), 
                                      torch.Tensor(Y_tra).to(device))
        validation_dataset = TensorDataset(torch.Tensor(X_val).to(device), 
                                     torch.Tensor(x1_val).to(device), 
                                     torch.Tensor(x2_val).to(device), 
                                     torch.Tensor(both_emb_val).to(device), 
                                     torch.Tensor(site_weight_val).to(device), 
                                     torch.Tensor(Y_val).to(device))
        test_dataset = TensorDataset(torch.Tensor(X_test).to(device), 
                                     torch.Tensor(x1_test).to(device), 
                                     torch.Tensor(x2_test).to(device), 
                                     torch.Tensor(both_emb_test).to(device), 
                                     torch.Tensor(site_weight_test).to(device), 
                                     torch.Tensor(Y_test).to(device))
        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)
        validation_loader = DataLoader(validation_dataset, batch_size = batch_size, shuffle = False)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

        ## 3. model initilization
        model = CSPred(input_dim_seq = x1_train.shape[1], 
                       input_dim_phos = x2_train.shape[1], 
                       dropout_rate = dropout_rate).to(device)

        ## 4. set up optimizer and criterion
        optimizer = optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
        # scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, verbose=0, min_lr=0, patience=3)

        ## 5. training and validation loop
        best_val_auc = 0.0 ## best validation auc of this k-fold iteration
        best_epoch = 0
        counter = 0 ## early stopping parameter
        best_model_path = None
        
        for epoch in range(300):
            ### 5-1. training
            model.train()
            train_loss = 0.0
            for _, inputs_seq, inputs_phos, inputs_both_emb, inputs_site_weight, targets in train_loader:
                optimizer.zero_grad()
                outputs = model.forward(inputs_seq, inputs_phos, inputs_both_emb)
                
                criterion = nn.BCEWithLogitsLoss(weight=inputs_site_weight)
                loss = criterion(outputs.squeeze(), targets)
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            train_loss /= len(train_loader)

            ### 5-2. valiadation
            model.eval()
            valid_loss = 0.0
            val_outputs = []
            val_targets = []
            for _, inputs_seq, inputs_phos, inputs_both_emb, inputs_site_weight, targets in validation_loader:
                with torch.no_grad():
                    outputs = model.forward(inputs_seq, inputs_phos, inputs_both_emb)
                
                criterion = nn.BCEWithLogitsLoss(weight=inputs_site_weight)
                loss = criterion(outputs.squeeze(), targets)
                valid_loss += loss.item()
                
                val_outputs.extend(np.atleast_1d(outputs.squeeze().detach().cpu().numpy()))
                val_targets.extend(np.atleast_1d(targets.cpu().numpy()))
            
            valid_loss /= len(test_loader)
            val_probas = torch.sigmoid(torch.tensor(val_outputs)).detach().cpu().numpy()
            
            # scheduler.step(valid_loss)
            
            val_auc = roc_auc_score(val_targets, val_probas)
            print('No. {} Num: {} Epoch: {} Train Loss: {:.4f} Val Loss: {:.4f} Val AUC: {:.4f}'.format(a+1, num, epoch+1, train_loss, valid_loss, val_auc))

            ### 5-3. early stopping
            if val_auc - best_val_auc > min_delta:
                best_val_auc = val_auc
                counter = 0
                best_epoch = epoch
                
                if best_model_path is not None:
                    os.remove(best_model_path)
                    os.remove(best_model_path.replace("model", "score") + ".npy")
                    os.remove(best_model_path.replace("model", "label") + ".npy")
            
                best_model_path = f"{path}/{a+1}_{num}.model.ep{best_epoch + 1}"
                torch.save(model.to('cpu'), best_model_path)
                model.to(device)
                
                np.save(best_model_path.replace("model", "score") + ".npy", val_probas)
                np.save(best_model_path.replace("model", "label") + ".npy", val_targets)
                
            else:
                counter += 1

            if counter > patience:
                break

        ## 5. test
        model = torch.load(best_model_path).to(device)
        model.eval()
        test_outputs = []
        Y_test = []
        for _, inputs_seq, inputs_phos, inputs_both_emb, inputs_site_weight, targets in test_loader:
            with torch.no_grad():
                outputs = model.forward(inputs_seq, inputs_phos, inputs_both_emb)

            test_outputs.extend(np.atleast_1d(outputs.squeeze().detach().cpu().numpy()))
            Y_test.extend(np.atleast_1d(targets.cpu().numpy()))

        np.save(best_model_path.replace("model", "test_score") + ".npy", test_outputs)
        np.save(best_model_path.replace("model", "test_label") + ".npy", Y_test)

        predict_x = torch.sigmoid(torch.tensor(test_outputs)).detach().cpu().numpy()
        rocauc = roc_auc_score(Y_test, predict_x)
        accuracy = accuracy_score(Y_test, np.array([i >= 0.5 for i in predict_x], dtype = "int"))
        f1 = f1_score(Y_test, np.array([i >= 0.5 for i in predict_x], dtype = "int"))
        precision, recall, _ = precision_recall_curve(Y_test, predict_x)
        auprc = auc(recall,precision)
        mcc = matthews_corrcoef(Y_test, np.array([i >= 0.5 for i in predict_x], dtype = "int"))
        
        if rocauc > best_auc:
            best_auc = rocauc
            best_model = num
        
        auc_all += rocauc
        accuracy_all += accuracy
        f1_all += f1
        auprc_all += auprc
        mcc_all += mcc
    
    auc_all = auc_all/5
    accuracy_all = accuracy_all/5
    f1_all = f1_all/5
    auprc_all = auprc_all/5
    mcc_all = mcc_all/5
        
    return auc_all, accuracy_all, f1_all, auprc_all, mcc_all, best_model
    