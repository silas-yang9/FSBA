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


METHODS = [
    'F2BA',
    'AID',
    'ITD',
    'RAHGD',
    'baselineRAHGD',
    'IFSBA'  # The newly added IFSBA method
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="mnist", choices=["mnist", "fashion"])
    parser.add_argument('--train_size', type=int, default=50000)
    parser.add_argument('--val_size', type=int, default=10000)
    parser.add_argument('--pretrain', type=int,  default=0, choices=[0,1], help='whether to create data and pretrain on valset')
    parser.add_argument('--epochs', type=int, default=5000)
    parser.add_argument('--iterations', type=int, default=10, help='T')
    parser.add_argument('--K', type=int, default=10, help='k')
    parser.add_argument('--data_path', default='~/Data', help='where to save data')
    parser.add_argument('--model_path', default='./save_data_cleaning', help='where to save model')
    parser.add_argument('--noise_rate', type=float, default=0.25)
    parser.add_argument('--x_lr', type=float, default=0.01)
    parser.add_argument('--xhat_lr', type=float, default=0.01)
    parser.add_argument('--w_lr', type=float, default=100)
    parser.add_argument('--theta1', type=float, default=0.95)
    parser.add_argument('--theta2', type=float, default=0.95)
    parser.add_argument('--eps', type=float, default=0.0001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--alg', type=str, default='F2BA', choices=METHODS)
    parser.add_argument('--lmbd', type=float, default=10.0)
    parser.add_argument('--M', type=float, default=10.0, help='New hyperparameter M') #cubic newton parameter
    
    # Advanced options for IFSBA (Cubic Regularization with Chebyshev)
    parser.add_argument('--use_cubic', type=int, default=1, choices=[0,1], help='Use cubic regularization for w update')
    parser.add_argument('--cheb_K', type=int, default=10, help='Chebyshev polynomial order for Hessian approximation')
    parser.add_argument('--l_est', type=float, default=100.0, help='Estimated upper bound of Hessian eigenvalues')
    parser.add_argument('--mu_est', type=float, default=0.01, help='Estimated lower bound of Hessian eigenvalues')
    parser.add_argument('--cubic_iters', type=int, default=5, help='Max iterations for cubic subproblem')
    parser.add_argument('--hessian_q', type=int, default=10, help='Number of Hessian CG iterations for RAHGD')

    # RAHGD specific parameters
    parser.add_argument('--rahgd_theta', type=float, default=0.5, help='Momentum parameter for RAHGD')
    parser.add_argument('--rahgd_B', type=float, default=0.1, help='Restart parameter B')
    parser.add_argument('--rahgd_r', type=float, default=0.01, help='Perturbation radius r')
    parser.add_argument('--rahgd_perturbation', type=int, default=1, choices=[0,1], help='Use perturbation (0=no, 1=yes)')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    return args


def get_data(args):

    data = {
        'mnist': datasets.MNIST,
        'fashion': datasets.FashionMNIST,
    }

    trainset = data[args.dataset](root=args.data_path,
                                  train=True,
                                  download=True)
    testset  = data[args.dataset](root=args.data_path,
                                  train=False,
                                  download=True)

    indices = torch.randperm(len(trainset))

    train_x = trainset.data[indices[:args.train_size]] / 255.
    val_x   = trainset.data[indices[args.train_size:args.train_size+args.val_size]] / 255.
    test_x  = testset.data / 255.

    targets = trainset.targets if args.dataset in ["mnist", "fashion"] else torch.LongTensor(trainset.targets) 
    train_y = targets[indices[:args.train_size]]
    val_y   = targets[indices[args.train_size:args.train_size+args.val_size]]
    test_y  = torch.LongTensor(testset.targets)

    num_classes = test_y.unique().shape[0]
    assert val_y.unique().shape[0] == num_classes

    ### poison training data with noise rate = args.noise_rate
    num_noisy = int(args.train_size * args.noise_rate)
    rand_indices = torch.randperm(args.train_size)
    noisy_indices = rand_indices[:num_noisy]
    noisy_y = torch.randint(num_classes, size=(num_noisy,))
    old_train_y = train_y.data.clone()
    train_y.data[noisy_indices] = noisy_y.data

    # normalizing inputs to mean 0 and std 1.
    mean = train_x.unsqueeze(1).mean([0,2,3])
    std  = train_x.unsqueeze(1).std([0,2,3])

    trainset = ( torch.flatten((train_x  - mean)/(std+1e-4), start_dim=1), train_y )
    valset   = ( torch.flatten((val_x    - mean)/(std+1e-4), start_dim=1), val_y   )
    testset = ( torch.flatten((test_x  - mean)/(std+1e-4), start_dim=1), test_y )

    return trainset, valset, testset, old_train_y

### initialize a linear model

def get_model(in_features, out_features, device):
    x = torch.zeros(out_features, in_features+1, requires_grad=True, device=device)

    weight = torch.empty((out_features, in_features))
    bias = torch.empty(out_features)
    nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weight)
    bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
    nn.init.uniform_(bias, -bound, bound)

    x[:,:in_features].data.copy_(weight.clone().to(device))
    x[:, -1].data.copy_(bias.clone().to(device))
    return x

def model_forward(x, inputs):
    in_features = 28*28
    A = x[:,:in_features] # (out_features, in_features)
    b = x[:,-1] # (out_features,)
    y = inputs.mm(A.t()) + b.view(1,-1)
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
### Chebyshev polynomial approximation for Hessian-vector products

def hessian_vector_product_chebyshev(v, hessian_func, K, l_est, mu_est):
    """
    Approximate H @ v using Chebyshev polynomial, where H is the Hessian.
    
    Args:
        v: Input vector
        hessian_func: Function that computes exact Hessian-vector product H @ u
        K: Chebyshev polynomial order
        l_est: Estimated upper bound of Hessian eigenvalues
        mu_est: Estimated lower bound of Hessian eigenvalues
    
    Returns:
        Approximation of H @ v
    """
    device = v.device
    
    # Normalize to [mu1, l1] = [mu_est/(2*l_est), 0.5]
    mu1 = mu_est / (2 * l_est)
    l1 = 0.5
    
    # Chebyshev parameters
    p1 = 2 / (l1 - mu1)
    p2 = (l1 + mu1) / (l1 - mu1)
    p3 = (math.sqrt(mu1 / l1) - 1) / (math.sqrt(mu1 / l1) + 1)
    c = 2 / math.sqrt(l1 * mu1)
    
    # T_0 = v
    T_prev = v.clone()
    
    # Compute H @ v / (2 * l_est)
    Hv = hessian_func(v) / (2 * l_est)
    
    # T_1 = p1 * (H @ v) - p2 * v
    T_curr = p1 * Hv - p2 * v
    
    # Initialize result: c/2 * T_0 + c*p3 * T_1
    result = c / 2 * T_prev
    c = c * p3
    result = result + c * T_curr
    
    # Chebyshev iterations k=2 to K-1
    for k in range(2, K):
        c = c * p3
        
        # Compute H @ T_curr / (2 * l_est)
        Hv = hessian_func(T_curr) / (2 * l_est)
        
        # T_next = 2 * p1 * (H @ T_curr) - 2 * p2 * T_curr - T_prev
        T_next = 2 * p1 * Hv - 2 * p2 * T_curr - T_prev
        
        # Accumulate result
        result = result + c * T_next
        
        # Shift for next iteration
        T_prev = T_curr
        T_curr = T_next
        
        # Memory management
        if k % 5 == 0:
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Final scaling
    result = result / (2 * l_est)
    
    return result


def cubic_newton_step_chebyshev(grad, hessian_func, M, K, l_est, mu_est,
                                max_iters=10, tol=1e-4):

    device = grad.device
    gnorm = grad.norm()

    # If gradient tiny → no movement
    if gnorm < 1e-6:
        return torch.zeros_like(grad)

    # ============================================================
    # 1. Analytical cubic step (for large-gradient regime)
    # ============================================================
    if gnorm >= 0.01:   # threshold 

        # Compute Hg using Chebyshev approx
        Hg = hessian_vector_product_chebyshev(
            grad, hessian_func, K, l_est, mu_est
        )

        gHg = (grad * Hg).sum()           # scalar g^T H g

        # ---- Analytical solution for Rc ----
        # Rc = - (gHg / (M ||g||^2)) + sqrt( (gHg/(M||g||^2))^2 + 2||g||/M )
        alpha = gHg / (M * (gnorm ** 2) + 1e-12)
        Rc = -alpha + torch.sqrt(alpha * alpha + 2 * gnorm / M)

        # Cauchy step
        s = -Rc * grad / (gnorm + 1e-12)
        return s

    # ============================================================
    # 2. Small gradient region: Iterative cubic solver
    # ============================================================
    s = torch.zeros_like(grad)
    lr = 0.1

    for i in range(max_iters):

        Hs = hessian_vector_product_chebyshev(
            s, hessian_func, K, l_est, mu_est
        )

        s_norm = s.norm()
        if s_norm < 1e-10:
            cubic_grad = grad + Hs
        else:
            cubic_grad = grad + Hs + (M/2) * s_norm * s

        if cubic_grad.norm() < tol:
            break

        s = s - lr * cubic_grad

    return s


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

def IFSBA(args, x, w, trainset, valset, testset, clean_indices):
    """
    IFSBA: Inexact Fully Second-Order Bilevel Approximation Algorithm
    
    Key components:
    - Inner loop: AGD (Accelerated Gradient Descent) for x
    - Outer loop: Cubic Regularization with Chebyshev approximation for w
    - Hessian updated at EVERY iteration (no lazy updates)
    """
    xhat = copy.deepcopy(x)
    total_time = 0.0
    stats = []
    
    for epoch in trange(args.epochs):
        t0 = time.time()
        
        # ============================================
        # Inner optimization: AGD for x and xhat
        # ============================================
        x = AGD_x(args, x, w, trainset, valset)
        xhat = AGD_xhat(args, xhat, w, trainset)
        
        x.requires_grad_(True)
        xhat.requires_grad_(True)
        w.requires_grad_(True)
       
        # ============================================
        # Outer optimization: Cubic regularization
        # ============================================
        
        # Compute gradient of w (computed at every iteration)
        _, _, gw_minus_gw_k = g_x_xhat_w(x, xhat, w, trainset)
        w_grad = args.lmbd * gw_minus_gw_k
        
        
        # Capture current x, xhat for Hessian computation
        x_snapshot = x.detach().clone().requires_grad_(True)
        xhat_snapshot = xhat.detach().clone().requires_grad_(True)
        
        def w_hessian_vector_product(v):
            """
            Compute Hessian of outer objective w.r.t. w multiplied by vector v.
            Uses the formula: H @ v = d/dw (grad_w^T @ v)
            """
            w_temp = w.detach().clone().requires_grad_(True)
            
            # Compute first-order gradient w.r.t. w WITH create_graph=True
            _, _, gw_tmp = g_x_xhat_w(x_snapshot, xhat_snapshot, w_temp, trainset, 
                                      retain_graph=True, create_graph=True)
            gw = args.lmbd * gw_tmp
            
            # Compute Hessian-vector product: H @ v = d/dw (gw^T @ v)
            Hv = torch.autograd.grad(
                outputs=gw, 
                inputs=w_temp, 
                grad_outputs=v,
                retain_graph=True,
                create_graph=False,
                allow_unused=True
            )[0]
            
            if Hv is None:
                # If no gradient, return zero vector
                Hv = torch.zeros_like(v)
            
            return Hv.detach()
        
        # ============================================
        # Cubic regularization update
        # ============================================
        if args.use_cubic:
            # Solve cubic subproblem using current Hessian
            # min_s  g^T s + 1/2 s^T H s + M/6 ||s||^3
            w_step = cubic_newton_step_chebyshev(
                grad=w_grad,
                hessian_func=w_hessian_vector_product,
                M=args.M,
                K=args.cheb_K,
                l_est=args.l_est,
                mu_est=args.mu_est,
                max_iters=args.cubic_iters,
                tol=1e-4
            )
            
            # Update w with cubic step
            with torch.no_grad():
                w.data = w.data + w_step
        else:
            # Fallback: simple first-order gradient descent
            with torch.no_grad():
                w.data = w.data - args.w_lr * w_grad
        
        # Clamp w to [0, 1]
        w.data.clamp_(0.0, 1.0)
        
        t1 = time.time()
        total_time += t1 - t0

        # ============================================
        # Evaluation
        # ============================================
        test_loss, test_acc = evaluate(x, testset)
        torch.cuda.empty_cache()
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc))
        
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | time {total_time:6.2f} | f1 {f1[2]:4.2f}")
    
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

def RAHGD(args, x, w, trainset, valset, testset, clean_indices):
    """
    (Perturbed) Restarted Accelerated HyperGradient Descent (P)RAHGD
    """
    outer_loss = lambda x, w: f(x[0], w[0], valset)
    inner_loss = lambda x, w, d: g(x[0], w[0], d)
    inner_opt_cg = hg.GradientDescent(inner_loss, 1., data_or_iter=trainset)
    
    # Parameters
    # Note: AGD_xhat uses args.theta2 internally, so we don't need separate alpha/beta parameters
    eta = args.w_lr  # Step size for outer loop
    theta = args.rahgd_theta
    B = args.rahgd_B
    r = args.rahgd_r
    perturbation = args.rahgd_perturbation
    K = args.iterations  # Iteration threshold K
    T_cg = args.hessian_q  # CG iterations
    # Note: AGD_xhat uses args.K internally, so no need for separate T_agd variable
    
    # Initialize (Algorithm step 2)
    # Note: In pseudocode, x is outer variable (w in code), y is inner variable (x in code)
    k = 0  # Inner iteration counter
    t = 0  # Restart counter
    w_t_prev = w.clone()  # x_{t,-1} in pseudocode (outer variable)
    x_t_prev = x.clone()  # y_{t,-1} in pseudocode (inner variable)
    
    # y_{0,-1} = AGD(g(x_{0,-1}, ·), 0, T_{0,-1}, α, β)
    # Use AGD_xhat which only optimizes g (not f)
    y_t_prev = AGD_xhat(args, torch.zeros_like(x), w, trainset)
    
    # v_{0,-1} = y_{0,-1} (will be updated by CG)
    v_t_prev = y_t_prev.clone()
    
    total_time = 0.0
    stats = []
    
    # Store w (outer variable) history for restart condition
    w_history = [w_t_prev.clone()]
    
    for epoch in trange(args.epochs):
        t0 = time.time()
        
        # Main loop: while k < K (Algorithm step 3)
        if k < K:
            # Step 4: Momentum step
            # w_{t,k} = x_{t,k} + (1 - θ)(x_{t,k} - x_{t,k-1})
            # Note: x in pseudocode is outer variable (w in code)
            if k == 0:
                w_tk = w_t_prev.clone()
            else:
                w_tk = w + (1 - theta) * (w - w_t_prev)
            
            # Step 5: AGD step
            # y_{t,k} = AGD(g(w_{t,k}, ·), y_{t,k-1}, T_{t,k}, α, β)
            # Use AGD_xhat which only optimizes g (not f)
            y_tk = AGD_xhat(args, y_t_prev, w_tk, trainset)
            
            # Step 6: CG step
            # v_{t,k} = CG(∇²_{yy}g(w_{t,k}, y_{t,k}), ∇_y f(w_{t,k}, y_{t,k}), T'_{t,k}, v_{t,k-1})
            # Use hypergrad's CG to compute hypergradient
            # CG solves the linear system and computes the hypergradient directly
            # Since w_tk is not a leaf tensor, we need to use retain_grad() or get gradient from return value
            w_tk.requires_grad_(True)
            w_tk.retain_grad()  # Retain gradient for non-leaf tensor
            outer_loss_cg = lambda x, w: f(x[0], w[0], valset)
            
            # Get hypergradient directly from CG return value (more reliable than .grad for non-leaf tensors)
            hypergrads = hg.CG([y_tk], [w_tk], T_cg, inner_opt_cg, outer_loss_cg, 
                               stochastic=False, set_grad=True, tol=1e-12)
            
            # The hypergradient: u_{t,k} = ∇_x f(w_{t,k}, y_{t,k}) - ∇²_{xy}g(w_{t,k}, y_{t,k})v_{t,k}
            # Use return value if available, otherwise try .grad (with retain_grad it should work)
            if hypergrads and hypergrads[0] is not None:
                u_tk = hypergrads[0].clone()
            elif w_tk.grad is not None:
                u_tk = w_tk.grad.clone()
            else:
                # Fallback: compute gradient manually if both fail
                raise RuntimeError("Failed to compute hypergradient in RAHGD")
            
            # For v_tk, we don't need it explicitly since CG computes the hypergradient directly
            # But we keep v_t_prev for potential use in restart
            v_tk = v_t_prev.clone()  # Placeholder, not used in computation
            
            # Step 8: Update x (outer variable)
            # x_{t,k+1} = w_{t,k} - η u_{t,k}
            w_new = w_tk - eta * u_tk
            
            # Step 9: k ← k + 1
            k += 1
            
            # Store for restart condition
            w_history.append(w_new.clone())
            
            # Step 10: Restart condition
            # if ∑_{i=0}^{k-1} ||x_{t,i+1} - x_{t,i}||² > B²
            s_sum = 0.0
            for i in range(k):
                if i + 1 < len(w_history):
                    s_sum += (w_history[i+1] - w_history[i]).norm().item() ** 2
            
            if s_sum > B ** 2:
                # Step 11: v_{t+1,-1} = v_{t,k}
                v_t_prev = v_tk.clone()
                
                # Step 12: Perturbation step
                if perturbation == 1:
                    # x_{t+1,0} = x_{t,k} + ξ with ξ ~ Unif(𝔹(r))
                    xi = torch.rand_like(w_new) * 2 * r - r  # Uniform in [-r, r]
                    xi = xi / (xi.norm() + 1e-12) * (torch.rand(1).item() * r)  # Scale to ball of radius r
                    w_new = w_new + xi
                # else: x_{t+1,0} = x_{t,k} (already w_new)
                
                # Step 13: x_{t+1,-1} = x_{t+1,0}
                w_t_prev = w_new.clone()
                
                # Step 14: k ← 0, t ← t + 1
                k = 0
                t += 1
                w_history = [w_t_prev.clone()]
                
                # Step 15: y_{t,-1} = AGD(g(x_{t,-1}, ·), 0, T_{t,-1}, α, β)
                # Update inner variable given new outer variable
                y_t_prev = AGD_xhat(args, torch.zeros_like(x), w_t_prev, trainset)
                # Also update x (inner variable) to match y_t_prev
                x = y_t_prev.clone()
            else:
                # No restart: update for next iteration
                w_t_prev = w.clone()
                w = w_new.clone()
                y_t_prev = y_tk.clone()
                v_t_prev = v_tk.clone()
                # Update inner variable x to match y_tk
                x = y_tk.clone()
        else:
            # k >= K: should not happen in this structure, but handle it
            break
        
        # Update w for clamping
        w.data.clamp_(0.0, 1.0)
        
        t1 = time.time()
        total_time += t1 - t0
        
        # Evaluation
        test_loss, test_acc = evaluate(x, testset)
        f1 = evaluate_importance_f1(w, clean_indices)
        stats.append((total_time, test_loss, test_acc))
        print(f"[info] epoch {epoch:5d} | te loss {test_loss:6.4f} | te acc {test_acc:4.2f} | time {total_time:6.2f} | k={k} t={t} | w-min {w.min().item():4.2f} w-max {w.max().item():4.2f}")
    
    return stats


def baselineRAHGD(args, x, w, trainset, valset, testset, clean_indices):
    """
    Baseline RAHGD in this environment: use the same RAHGD pipeline with perturbation disabled.
    """
    baseline_args = copy.deepcopy(args)
    baseline_args.rahgd_perturbation = 0
    return RAHGD(
        args=baseline_args,
        x=x,
        w=w,
        trainset=trainset,
        valset=valset,
        testset=testset,
        clean_indices=clean_indices
    )


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
        elif args.alg == 'IFSBA':
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_lmbd{args.lmbd}_sd{args.seed}"
        else:
            save_path = f"./{args.model_path}/{args.dataset}_{args.alg}_k{args.iterations}_xlr{args.x_lr}_wlr{args.w_lr}_xhatlr{args.xhat_lr}_sd{args.seed}"

        torch.save(stats, save_path)
