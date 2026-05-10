import argparse
import copy
import hypergrad as hg # hypergrad package
import math
import numpy as np
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from functorch import hessian
from sklearn.datasets import load_svmlight_file
from torch.utils.data import Dataset, random_split


from torchvision import datasets

from tqdm import trange


################################################################################
#
#  Bilevel Optimization
#
#  min_{x,w} f(x, w)
#  s.t. x = argmin_x g(x, w)
#
#  here: f(x, w) is on valset
#        g(x, w) is on trainset
#
#  f_x = df/dx
#  f_w = df/dw
#  g_x = dg/dx
#  g_w = dg/dw
#
################################################################################

""" METHODS = [
   'F2BA',
    'AID',
    'ITD'
] """

METHODS = [
    'F2BA',
    'AID',
    'ITD',
    'LFSBA'  # The newly added FSBA method
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="breast", choices=["breast", "fashion"])
    parser.add_argument('--train_ratio', type=int, default=0.7)
    parser.add_argument('--val_ratio', type=int, default=0.2)
    parser.add_argument('--pretrain', type=int,  default=0, choices=[0,1], help='whether to create data and pretrain on valset')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--iterations', type=int, default=10, help='T')
    parser.add_argument('--K', type=int, default=10, help='k')
    parser.add_argument('--data_path', default='~/Data', help='where to save data')
    parser.add_argument('--data2_path', default='breast.txt', help='where to save data')
    parser.add_argument('--model_path', default='./save_data_cleaning', help='where to save model')
    parser.add_argument('--noise_rate', type=float, default=0.5)
    parser.add_argument('--x_lr', type=float, default=0.01)
    parser.add_argument('--xhat_lr', type=float, default=0.01)
    parser.add_argument('--w_lr', type=float, default=100)
    parser.add_argument('--theta1', type=float, default=0.95)
    parser.add_argument('--theta2', type=float, default=0.95)
    parser.add_argument('--m', type=int, default=10)
    parser.add_argument('--eps', type=float, default=0.0001)
   
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='F2BA', choices=METHODS)
    parser.add_argument('--lmbd', type=float, default=10.0)
    parser.add_argument('--M', type=float, default=10.0, help='New hyperparameter M') #cubic newton parameter

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def get_data(args):
    
    data_path = args.data2_path
    X, y = load_svmlight_file(data_path)
    
    
    X = torch.tensor(X.toarray(), dtype=torch.float32)  
    y = torch.tensor(y, dtype=torch.long)               

    
    class LibSVMDataSet(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]

    
    dataset = LibSVMDataSet(X, y)

    
    train_size = int(len(dataset) * args.train_ratio)   
    val_size = int(len(dataset) * args.val_ratio)       
    test_size = len(dataset) - train_size - val_size    

    indices = torch.randperm(len(dataset)).tolist()     

    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]

    train_x = X[train_indices]
    train_y = y[train_indices]
    val_x = X[val_indices]
    val_y = y[val_indices]
    test_x = X[test_indices]
    test_y = y[test_indices]

    
    train_y = (train_y == 4).long()  # If the label is 4, map it to 1; otherwise, map it to 0
    val_y = (val_y == 4).long()
    test_y = (test_y == 4).long()

    # noise
    num_noisy = int(train_size * args.noise_rate)
    rand_indices = torch.randperm(train_size)[:num_noisy]
    
    noisy_y = 1 - train_y[rand_indices]
    old_train_y = train_y.clone()
    train_y[rand_indices] = noisy_y

    
    mean = train_x.mean(dim=0)
    std = train_x.std(dim=0)

    train_x = (train_x - mean) / (std + 1e-4)
    val_x = (val_x - mean) / (std + 1e-4)
    test_x = (test_x - mean) / (std + 1e-4)

    
    trainset = (train_x, train_y)
    valset = (val_x, val_y)
    testset = (test_x, test_y)
    return trainset, valset, testset, old_train_y


### initialize a linear model

def get_model(in_features, out_features, device):
    
    x = torch.zeros(out_features, in_features + 1, requires_grad=True, device=device)

    
    weight = torch.empty((out_features, in_features))
    bias = torch.empty(out_features)
    
    
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    
    
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    
    x[:, :in_features].data.copy_(weight.clone().to(device))
    x[:, -1].data.copy_(bias.clone().to(device))
    
    return x

def model_forward(x, inputs):
    in_features = x.size(1) - 1  
    A = x[:, :in_features]       
    b = x[:, -1]                 
    y = inputs.mm(A.t()) + b.view(1, -1)  
    return y
### original f, g, and gradients

def f(x, w, dataset):
    data_x, data_y = dataset
    y = model_forward(x, data_x)
    loss = F.cross_entropy(y, data_y)
    return loss

def g(x, w, dataset):
    data_x, data_y = dataset
    y = model_forward(x, data_x)
    loss = F.cross_entropy(y, data_y, reduction='none')
    loss = (loss * torch.clip(w, 0, 1)).mean() + 0.001 * x.norm(2).pow(2)
    return loss

def g_x(x, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def g_w(x, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, w,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def g_x_xhat_w(x, xhat, w, dataset, retain_graph=False, create_graph=False):
    loss = g(x, w, dataset) - g(xhat.detach(), w, dataset)
    grad = torch.autograd.grad(loss, [x, w],
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return loss, grad[0], grad[1]


def f_x(x, w, dataset, retain_graph=False, create_graph=False):
    loss = f(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad


### second-order information for x and y

def g_xx_hessian(x,w,dataset):

    x.requires_grad_(True)
    
    loss = g(x, w, dataset)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=True,
                               create_graph=True)[0]
    
    
    grad_flat = grad.contiguous().view(-1)
    
    
    n = x.numel()
    hessian_matrix = torch.zeros((n, n), device=x.device)
    
    
    for i in range(n):
        try:
            
            grad_element = grad_flat[i]
            
            
            hessian_row = torch.autograd.grad(grad_element, x, retain_graph=True)[0]
            
            
            hessian_matrix[i] = hessian_row.contiguous().view(-1)
        except Exception as e:
            print(f"计算第 gxx{i} 行 Hessian 时出错: {e}")
    return hessian_matrix

def f_xx_hessian(x, w, dataset):
    
    x.requires_grad_(True)
    loss = f(x, w, dataset)
    
    
    grad = torch.autograd.grad(loss, x, retain_graph=True, create_graph=True)[0]
    
    
    grad_flat = grad.contiguous().view(-1)
    
    
    n = x.numel()
    hessian_matrix = torch.zeros((n, n), device=x.device)
    
    
    for i in range(n):
        try:
            
            grad_element = grad_flat[i]
            
            
            hessian_row = torch.autograd.grad(grad_element, x, retain_graph=True)[0]
            
            
            hessian_matrix[i] = hessian_row.contiguous().view(-1)
        except Exception as e:
            print(f"计算第 fxx{i} 行 Hessian 时出错: {e}")
    
    return hessian_matrix


def jacobian_g_xw(xhat, w, trainset):
    
    xhat.requires_grad_(True)
    
    
    loss = g(xhat, w, trainset)
    g_w_result = torch.autograd.grad(loss, w, create_graph=True)[0]
    
    
    g_w_flat = g_w_result.reshape(-1)
    
    
    jacobian = torch.zeros((w.numel(), xhat.numel()), device=xhat.device)
    
    
    for i in range(g_w_flat.numel()):
        
        g_w_element = g_w_flat[i]
        
        try:
            
            grad_x = torch.autograd.grad(g_w_element, xhat, retain_graph=True)[0]
            
            
            jacobian[i] = grad_x.reshape(-1)
        except Exception as e:
            print(f"Error calculating Jacobian matrix for row {i}: {e}")
    
    return jacobian


###AGD,momentum is dynamic,it is difficult to compute the condition number
###  AGD_x
def AGD_x(args, y_init, w_val, dataset_1, dataset_2):
    
    y = y_init.clone().detach()
    v = y.clone().detach().requires_grad_(True)  

    for k in range(args.K):
        
        gx = g_x(v, w_val, dataset_1)  # compute g(x)
        fx = f_x(v, w_val, dataset_2)

        grad = fx + args.lmbd * gx
        
        
        y_new = v - args.x_lr * grad
        v = y_new + args.theta1 * (y_new - y)
        y = y_new.clone().detach()  

    return y

### AGD_xhat


def AGD_xhat(args, y_init, w_val, dataset_1):
    
    y = y_init.clone().detach()
    v = y.clone().detach().requires_grad_(True)  

    for k in range(args.K):
        
        gx = g_x(v, w_val, dataset_1)
        grad = args.lmbd * gx  

        
        y_new = v - args.xhat_lr * grad
        v = y_new + args.theta2 * (y_new - y)
        y = y_new.clone().detach()  

    return y

### cubic newton step
def cubic_newton_step_torch(g, A, H, B=None, eps=1e-6, max_iters=30):
    n = g.shape[0]

    device = g.device if g.is_cuda else 'cpu'
    
    if B is None:
        B = torch.eye(n, device=device)
        l2_norm_sqr = lambda x: x @ x
    else:
        l2_norm_sqr = lambda x: x @ B @ x

    def h(r, compute_derivative=False):
       
        try:
            ArB = A + H * r * B
            T = torch.linalg.solve(ArB, -g)
            T_norm = torch.sqrt(l2_norm_sqr(T))
            h_r = r - T_norm
            
            if compute_derivative:
                BT = B @ T
                ArB_inv_BT = torch.linalg.solve(ArB, BT)
                h_r_prime = 1 + H / T_norm * (ArB_inv_BT @ BT)
                return h_r, T_norm, T, h_r_prime
            
            return h_r, T_norm, T, None
        except torch.linalg.LinAlgError:
            return None, None, None, None
    
    try:
        # 1.  max_r
        max_r = torch.tensor(1.0, device=device)
        for _ in range(max_iters):
            h_r, T_norm, T, _ = h(max_r)
            if h_r is None:
                #return torch.zeros(n, 1, device=device), 0.0, 0.0, "linalg_error"
                return torch.zeros(n, device=device), 0.0, 0.0, "linalg_error"
            if h_r < -eps:
                max_r *= 2
            elif -eps <= h_r <= eps:
                return T, h_r, max_r, "success"
            else:
                break
        
        # 2. Use Newton iteration to solve h(r) = 0
        r = max_r
        for _ in range(max_iters):
            h_r, T_norm, T, h_r_prime = h(r, compute_derivative=True)
            if h_r is None or h_r_prime is None:
                #return torch.zeros(n, 1, device=device), 0.0, 0.0, "linalg_error"
                return torch.zeros(n, device=device), 0.0, 0.0, "linalg_error"
            if -eps <= h_r <= eps:
                return T, h_r, r, "success"
            r -= h_r / h_r_prime
    except:
        return torch.zeros(n, 1, device=device), 0.0, 0.0, "error"
    
    return torch.zeros(n, 1, device=device), 0.0, 0.0, "iterations_exceeded"



### Define evaluation metric

def evaluate(x, testset):
    with torch.no_grad():
        test_x, test_y = testset  
        y = model_forward(x, test_x)
        test_loss = F.cross_entropy(y, test_y).detach().item()
        test_acc = y.argmax(-1).eq(test_y).float().mean().detach().cpu().item()
        
    return test_loss, test_acc


def evaluate_importance_f1(w, clean_indices):
    with torch.no_grad():
        w_ = w.gt(0.5).float()
        TP = (w_ * clean_indices.float()).sum()
        recall = TP / (clean_indices.float().sum()+1e-4)
        precision = TP / (w_.sum()+1e-4)
        f1 = 2.0 * recall * precision / (recall + precision + 1e-4)
    return precision.cpu().item(), recall.cpu().item(), f1.cpu().item()


###############################################################################
#
# Bilevel Optimization Training Methods
#
###############################################################################

def simple_train(args, x, data_x, data_y, testset, tag='pretrain', regularize=False): # directly train on the dataset
    opt = torch.optim.SGD([x], lr=args.x_lr)
    n = data_x.shape[0]

    n_epochs = 5000
    final_test_loss = np.inf
    final_test_acc = 0.
    best_x = None

    for epoch in range(n_epochs):
        opt.zero_grad()
        y = model_forward(x, data_x)
        loss = F.cross_entropy(y, data_y)
        if regularize:
            loss += 0.001 * x.norm(2).pow(2)
        loss.backward()
        opt.step()

        test_loss, test_acc = evaluate(x, testset)
        
        if test_loss <= final_test_loss:
            final_test_loss = test_loss
            final_test_acc  = test_acc
            best_x = x.data.clone()
        print(f"[{tag}] epoch {epoch:5d} test loss {test_loss:10.4f} test acc {test_acc:10.4f}")
    
    return final_test_loss, final_test_acc, best_x

def F2BA(args, x, w, trainset, valset, testset, clean_indices):
    xhat = copy.deepcopy(x)

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []
    
    outer_opt = torch.optim.SGD([w], lr=args.w_lr)
    inner_opt = torch.optim.SGD([
        {'params': [x], 'lr': args.x_lr},
        {'params': [xhat], 'lr': args.xhat_lr}])

    for epoch in trange(args.epochs):

        xhat.data = x.data.clone()
        t0 = time.time()
        for it in range(args.iterations):
            inner_opt.zero_grad()
            gx = g_x(xhat, w, trainset)
            fx = f_x(x, w, valset)
            xhat.grad =  args.lmbd * gx
            x.grad = fx  + args.lmbd * gx 
            inner_opt.step()

        _, gx, gw_minus_gw_k = g_x_xhat_w(x, xhat, w, trainset)
        outer_opt.zero_grad()
        w.grad =  args.lmbd * gw_minus_gw_k 
        outer_opt.step()

        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc = evaluate(x, testset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats

def LFSBA(args, x, w, trainset, valset, testset, clean_indices):
    xhat = copy.deepcopy(x)

    total_time = 0.0
    n = trainset[0].shape[0]
    stats = []
    
    for epoch in trange(args.epochs):

        t0 = time.time()
        # inner objective
        # use AGD optimize x and xhat

        x = AGD_x(args, x, w, trainset, valset)
        xhat = AGD_xhat(args, xhat, w, trainset)

        x.requires_grad_(True)
        xhat.requires_grad_(True)
        w.requires_grad_(True)
       
        _, _, gw_minus_gw_k = g_x_xhat_w(x, xhat, w, trainset)
        
        w_grad =  args.lmbd * gw_minus_gw_k 
        
        if epoch % args.m == 0:
            # 
            g_xxhat = g_xx_hessian(xhat, w, trainset)
            g_xwhat = jacobian_g_xw(xhat, w, trainset)

            g_xx=g_xx_hessian(x, w, trainset)
            g_xw = jacobian_g_xw(x, w, trainset)


            # 
            g_star_generic = -torch.linalg.solve(g_xxhat, g_xwhat.T).T  

            # 
            f_xx = f_xx_hessian(x, w, valset)

            # 
            L_wx = args.lmbd * g_xw.T
            L_xx = args.lmbd * g_xx + f_xx

            # 
            L_generic = -torch.linalg.solve(L_xx, L_wx).T  

            # 
            w_hessian = L_generic @ L_wx - args.lmbd * g_star_generic @ g_xwhat.T

        T, _, _, _ = cubic_newton_step_torch(w_grad, w_hessian, args.M)
                
        w=w+T
        
        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        test_loss, test_acc = evaluate(x, testset)
        torch.cuda.empty_cache()
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f} | f1 {f1[2]:4.2f}")
    return stats

def AID(args, x, w, trainset, valset, testset,  clean_indices):
    outer_loss = lambda x, w: f(x[0], w[0], valset)
    inner_loss = lambda x, w, d: g(x[0], w[0], d)

    inner_opt = hg.GradientDescent(inner_loss, args.x_lr, data_or_iter=trainset)
    inner_opt_cg = hg.GradientDescent(inner_loss, 1., data_or_iter=trainset)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr)

    total_time = 0.0
    stats = []

    for epoch in trange(args.epochs):

        t0 = time.time()
        x_history = [[x]]
        for it in range(args.iterations):
            x_history.append(inner_opt(x_history[-1], [w], create_graph=False))

        outer_opt.zero_grad()
        hg.CG([x_history[-1][0]], [w], args.K, inner_opt_cg, outer_loss, stochastic=False, set_grad=True)
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        x.data = x_history[-1][0].data.clone()
        test_loss, test_acc = evaluate(x, testset)
        stats.append((total_time, test_loss, test_acc))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats


def ITD(args, x, w, trainset, valset, testset,  clean_indices):
    outer_loss = lambda x, w: f(x[0], w[0], valset)
    inner_loss = lambda x, w, d: g(x[0], w[0], d)

    inner_opt = hg.GradientDescent(inner_loss, args.x_lr, data_or_iter=trainset)
    outer_opt = torch.optim.SGD([w], lr=args.w_lr)

    total_time = 0.0
    stats = []

    for epoch in trange(args.epochs):

        momentum = torch.zeros_like(x) 
        t0 = time.time()
        x_history = [[x]]
        for it in range(args.iterations):
            x_history.append(inner_opt(x_history[-1], [w], create_graph=True))

        outer_opt.zero_grad()
        loss = outer_loss([x_history[-1][0]], [w])
        grad = torch.autograd.grad(loss, w)[0]
        w.grad = grad.data.clone()
        outer_opt.step()
        t1 = time.time()
        total_time += t1 - t0
        w.data.clamp_(0.0, 1.0)

        x.data = x_history[-1][0].data.clone()

        test_loss, test_acc = evaluate(x, testset)
        stats.append((total_time, test_loss, test_acc))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | time {total_time:6.2f} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    return stats


if __name__ == "__main__":
    args = parse_args()

    if args.pretrain == True: # preprocess data and pretrain a model on validation set
    # if True: # preprocess data and pretrain a model on validation set

        if not os.path.exists(args.data_path):
            os.makedirs(args.data_path)

        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)

        ### generate data
        trainset, valset, testset, old_train_y = get_data(args)
        torch.save((trainset, valset, testset, old_train_y),
                   os.path.join(args.data_path, f"{args.dataset}_data_cleaning.pt"))
        print(f"[info] successfully generated data to {args.data_path}/{args.dataset}_data_cleaning.pt")

        ### pretrain a model and save it
        n_feats = np.prod(*trainset[0].shape[1:])
        num_classes = trainset[1].unique().shape[-1]
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        trainset = (trainset[0].to(args.device), trainset[1].to(args.device))
        valset   = (valset[0].to(args.device),   valset[1].to(args.device))
        testset  = (testset[0].to(args.device),  testset[1].to(args.device))
        old_train_y = old_train_y.to(args.device)

        x = get_model(n_feats, num_classes, args.device)
        sd = x.data.clone()

        # lower bound (train on noisy train + valset)
        tmp_x = torch.cat([trainset[0], valset[0]], 0)
        tmp_y = torch.cat([trainset[1], valset[1]], 0)
        test_loss1, test_acc1, best_x1 = simple_train(args, x, tmp_x, tmp_y, testset, regularize=True)
        torch.save(best_x1.data.cpu().clone(),
                   os.path.join(args.model_path, f"{args.dataset}_pretrained.pt"))

        # a baseline: train on valset
        x.data.copy_(sd)
        test_loss2, test_acc2, best_x2 = simple_train(args, x, valset[0], valset[1], testset)
        torch.save(best_x2.data.cpu().clone(),
                   os.path.join(args.model_path, f"{args.dataset}_pretrained_val.pt"))

        # upper bound (train on correct train + valset)
        x.data.copy_(sd)
        tmp_x = torch.cat([trainset[0], valset[0]], 0)
        tmp_y = torch.cat([old_train_y, valset[1]], 0)
        test_loss3, test_acc3, best_x3 = simple_train(args, x, tmp_x, tmp_y, testset)
        torch.save(best_x3.data.cpu().clone(),
                   os.path.join(args.model_path, f"{args.dataset}_pretrained_trainval.pt"))

        print(f"[pretrained] noisy train + val   : test loss {test_loss1} test acc {test_acc1}")
        print(f"[pretrained] val                 : test loss {test_loss2} test acc {test_acc2}")
        print(f"[pretrained] correct train + val : test loss {test_loss3} test acc {test_acc3}")

        torch.save({
            "pretrain_test_loss": test_loss1,
            "pretrain_test_acc": test_acc1,
            "pretrain_val_test_loss": test_loss2,
            "pretrain_val_test_acc": test_acc2,
            "pretrain_trainval_test_loss": test_loss3,
            "pretrain_trainval_test_acc": test_acc3,
            }, os.path.join(args.model_path, f"{args.dataset}_pretrained.stats"))


    else: # load pretrained model on valset and then start model training
        trainset, valset, testset, old_train_y = torch.load(
                os.path.join(args.data_path, f"{args.dataset}_data_cleaning.pt"))
        args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        n_feats = np.prod(*trainset[0].shape[1:])
        num_classes = trainset[1].unique().shape[-1]

        trainset = (trainset[0].to(args.device), trainset[1].to(args.device))
        valset   = (valset[0].to(args.device),   valset[1].to(args.device))
        testset  = (testset[0].to(args.device),  testset[1].to(args.device))
        old_train_y = old_train_y.to(args.device)

        x = get_model(n_feats, num_classes, args.device)
        x.data.copy_(torch.load(os.path.join(args.model_path, f"{args.dataset}_pretrained.pt")).to(args.device))

        # load the pretrained model on validation set
        pretrained_stats = torch.load(
            os.path.join(args.model_path, f"{args.dataset}_pretrained.stats"))

        test_loss1 = pretrained_stats['pretrain_test_loss']
        test_loss2 = pretrained_stats['pretrain_val_test_loss']
        test_loss3 = pretrained_stats['pretrain_trainval_test_loss']
        test_acc1  = pretrained_stats['pretrain_test_acc']
        test_acc2  = pretrained_stats['pretrain_val_test_acc']
        test_acc3  = pretrained_stats['pretrain_trainval_test_acc']
        print(f"[pretrained] noisy train + val   : test loss {test_loss1} test acc {test_acc1}")
        print(f"[pretrained] val                 : test loss {test_loss2} test acc {test_acc2}")
        print(f"[pretrained] correct train + val : test loss {test_loss3} test acc {test_acc3}")

        test_loss, test_acc = evaluate(x, testset)
        print("original test loss ", test_loss, "original test acc ", test_acc)

        clean_indices = old_train_y.to(args.device).eq(trainset[1])
        w = torch.zeros(trainset[0].shape[0], requires_grad=True, device=args.device)
        w.data.add_(0.5)

        stats = eval(args.alg)(args=args,
                               x=x,
                               w=w,
                               trainset=trainset,
                               valset=valset,
                               testset=testset,
                               clean_indices=clean_indices)
        
        if args.alg == 'F2BA':
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_lmbd{args.lmbd}_sd{args.seed}"
        elif args.alg == 'LFSBA':
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_lmbd{args.lmbd}_sd{args.seed}"
        else:
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"

        torch.save(stats, save_path)