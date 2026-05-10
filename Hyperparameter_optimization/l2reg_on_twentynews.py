from itertools import repeat
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import argparse
import torch
import hypergrad as hg
import numpy as np
import math
import time
import os
import pickle
import copy
from tqdm import trange



class CustomTensorIterator:
    def __init__(self, tensor_list, batch_size, **loader_kwargs):
        self.loader = DataLoader(TensorDataset(*tensor_list), batch_size=batch_size, **loader_kwargs)
        self.iterator = iter(self.loader)

    def __next__(self, *args):
        try:
            idx = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.loader)
            idx = next(self.iterator)
        return idx


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=45, type=int, help='epoch numbers')
    parser.add_argument('--T', default=10, type=int, help='inner update iterations')
    parser.add_argument('--batch_size', type=int, default=5657)
    parser.add_argument('--val_size', type=int, default=5657)
    parser.add_argument('--eta', type=float, default=0.5, help='used in Hessian')
    parser.add_argument('--hessian_q', type=int, default=10, help='number of steps to approximate hessian')
    parser.add_argument('--alg', type=str, default='RAHGD', choices=['PRAHGD', 'RAHGD', 'AID', 'PAID', 'ITD', 'BA-CG', 'IFSBA', 'F2BA'])
    parser.add_argument('--training_size', type=int, default=5657)
    parser.add_argument('--inner_lr', type=float, default=100.0)
    parser.add_argument('--inner_mu', type=float, default=0.0)
    parser.add_argument('--outer_lr', type=float, default=100.0)
    parser.add_argument('--outer_mu', type=float, default=0.0)
    parser.add_argument('--save_folder', type=str, default='', help='path to save result')
    parser.add_argument('--model_name', type=str, default='', help='Experiment name')
    
    # IFSBA specific parameters
    parser.add_argument('--use_cubic', type=int, default=1, choices=[0,1], help='Use cubic regularization for IFSBA')
    parser.add_argument('--cheb_K', type=int, default=10, help='Chebyshev polynomial order when updating Hessian')
    parser.add_argument('--l_est', type=float, default=100.0, help='Estimated upper bound of Hessian eigenvalues')
    parser.add_argument('--mu_est', type=float, default=0.01, help='Estimated lower bound of Hessian eigenvalues')
    parser.add_argument('--cubic_iters', type=int, default=5, help='Max iterations for cubic subproblem when updating Hessian')
    parser.add_argument('--M', type=float, default=10.0, help='Cubic regularization parameter M')
    parser.add_argument('--m', type=int, default=10, help='Update Hessian every m epochs for IFSBA')
    parser.add_argument('--theta1', type=float, default=0.95, help='AGD momentum parameter for inner loop')
    parser.add_argument('--lmbd', type=float, default=1.0, help='Lambda parameter for IFSBA and F2BA')
    
    # F2BA specific parameters
    parser.add_argument('--f2ba_tau', type=float, default=0.01, help='Learning rate tau for y update in F2BA')
    parser.add_argument('--f2ba_alpha', type=float, default=0.01, help='Learning rate alpha for z update in F2BA')
    parser.add_argument('--f2ba_K', type=int, default=10, help='Number of inner iterations K for F2BA')
    
    args = parser.parse_args()


    if not args.save_folder:
        args.save_folder = './save_results'
    args.model_name = '{}_bs_{}_vbs_{}_olrmu_{}_{}_ilrmu_{}_{}_eta_{}_T_{}_hessianq_{}'.format(args.alg, 
                       args.batch_size, args.val_size, args.outer_lr, args.outer_mu, args.inner_lr, 
                       args.inner_mu, args.eta, args.T, args.hessian_q)
    args.save_folder = os.path.join(args.save_folder, args.model_name)
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    return args


def train_model(args):

    # Constant
    tol = 1e-12
    warm_start = True
    bias = False
    train_log_interval = 100
    val_log_interval = 1

    # Basic Setting 
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    cuda = True and torch.cuda.is_available()
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
    if cuda:
        torch.set_default_device('cuda')
    torch.set_default_dtype(torch.float32)

    # Functions 
    def frnp(x): return torch.from_numpy(x).cuda().float() if cuda else torch.from_numpy(x).float()
    def tonp(x, cuda=cuda): return x.detach().cpu().numpy() if cuda else x.detach().numpy()

    def train_loss(params, hparams, data):
        x_mb, y_mb = data
        # print(x_mb.size()) = torch.Size([5657, 130107])
        out = out_f(x_mb,  params)
        return F.cross_entropy(out, y_mb) + reg_f(params, *hparams)

    def val_loss(opt_params, hparams):
        x_mb, y_mb = next(val_iterator)
        # print(x_mb.size()) = torch.Size([5657, 130107])
        out = out_f(x_mb,  opt_params[:len(parameters)])
        val_loss = F.cross_entropy(out, y_mb)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y_mb.view_as(pred)).sum().item() / len(y_mb)

        val_losses.append(tonp(val_loss))
        val_accs.append(acc)
        return val_loss

    def reg_f(params, l2_reg_params, l1_reg_params=None):
        r = 0.5 * ((params[0] ** 2) * torch.exp(l2_reg_params.unsqueeze(1) * ones_dxc)).mean()
        if l1_reg_params is not None:
            r += (params[0].abs() * torch.exp(l1_reg_params.unsqueeze(1) * ones_dxc)).mean()
        return r

    def out_f(x, params):
        out = x @ params[0]
        out += params[1] if len(params) == 2 else 0
        return out

    def eval(params, x, y):
        out = out_f(x,  params)
        loss = F.cross_entropy(out, y)
        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        acc = pred.eq(y.view_as(pred)).sum().item() / len(y)

        return loss, acc

    
    # ============================================
    # Load offline TF-IDF 20news dataset
    # ============================================

    print("Loading offline TF-IDF 20news dataset...")
    pkl_path = "/root/scikit_learn_data/20news-bydate-tfidf.pkl"

    X, y, train_mask = pickle.load(open(pkl_path, "rb"))

    # split train/test using mask
    x_train = X[train_mask]
    x_test  = X[~train_mask]
    y_train = y[train_mask]
    y_test  = y[~train_mask]

    # split train/val
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.5, stratify=y_train
    )

    train_samples, n_features = x_train.shape
    val_samples, _ = x_val.shape
    test_samples, _ = x_test.shape
    n_classes = np.unique(y_train).shape[0]

    print('Dataset 20newsgroup (offline TF-IDF), train_samples=%i, val_samples=%i, test_samples=%i, n_features=%i, n_classes=%i'
        % (train_samples, val_samples, test_samples, n_features, n_classes))

    # convert scipy sparse → torch sparse
    def from_sparse(x):
        x = x.tocoo()
        values = x.data
        indices = np.vstack((x.row, x.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = x.shape
        return torch.sparse_coo_tensor(i, v, torch.Size(shape), dtype=torch.float32)

    if cuda:
        xs = [from_sparse(x).cuda() for x in (x_train, x_val, x_test)]
    else:
        xs = [from_sparse(x) for x in (x_train, x_val, x_test)]

    x_train, x_val, x_test = xs
    y_train = frnp(y_train).long()
    y_val   = frnp(y_val).long()
    y_test  = frnp(y_test).long()

    print("Torch tensors ready.")
    
    # torch.DataLoader has problems with sparse tensor on GPU
    iterators, train_list, val_list = [], [], []
    xmb_train, xmb_val, ymb_train, ymb_val = [], [], [], []

    for bs, x, y in [(len(y_train), x_train, y_train), (len(y_val), x_val, y_val)]:
        iterators.append(repeat([x, y]))
    train_iterator, val_iterator = iterators


    # Initialize parameters
    l2_reg_params = torch.zeros(n_features).requires_grad_(True)  # one hp per feature
    l1_reg_params = (0.*torch.ones(1)).requires_grad_(True)  # one l1 hp only (best when really low)
    #l2_reg_params = (-20.*torch.ones(1)).requires_grad_(True)  # one l2 hp only (best when really low)
    #l1_reg_params = (-1.*torch.ones(n_features)).requires_grad_(True)
    hparams = [l2_reg_params]
    # hparams: the outer variables (or hyperparameters)
    ones_dxc = torch.ones(n_features, n_classes)

    outer_opt = torch.optim.SGD(lr=args.outer_lr, momentum=args.outer_mu, params=hparams)
    # outer_opt = torch.optim.Adam(lr=0.01, params=hparams)

    params_history = []
    val_losses, val_accs = [], []
    test_losses, test_accs = [], []
    w = torch.zeros(n_features, n_classes).requires_grad_(True)
    parameters = [w]

    # params_history: the inner iterates (from first to last)
    if bias:
        b = torch.zeros(n_classes).requires_grad_(True)
        parameters.append(b)
 
    if args.inner_mu > 0:
        #inner_opt = hg.Momentum(train_loss, inner_lr, inner_mu, data_or_iter=train_iterator)
        inner_opt = hg.HeavyBall(train_loss, args.inner_lr, args.inner_mu, data_or_iter=train_iterator)
    else:
        inner_opt = hg.GradientDescent(train_loss, args.inner_lr, data_or_iter=train_iterator)
    inner_opt_cg = hg.GradientDescent(train_loss, 1., data_or_iter=train_iterator)

    total_time = 0
    calls_num = 0
    loss_acc_time_results = np.zeros((args.epochs+1, 4))
    test_loss, test_acc = eval(parameters, x_test, y_test)
    loss_acc_time_results[0, 0] = test_loss
    loss_acc_time_results[0, 1] = test_acc
    loss_acc_time_results[0, 2] = 0.0
    loss_acc_time_results[0, 3] = 0

    if args.alg == 'PRAHGD':
        r = 0.1
        hparams0 = hparams[0] + torch.rand_like(hparams[0]) * r    # used in PRAHGD
        k, s = 0, 0
    elif args.alg == 'RAHGD':
        hparams0 = hparams[0]
        k, s = 0, 0
    pk, huaT = 0, 3  # used in PAID
    cached_hessian_func = None  # Initialize for LFSBA
    for o_step in range(args.epochs):
        start_time = time.time()
        if args.alg == 'PRAHGD':
            inner_losses = []
            inner_theta = 0.005  # parameter of AGD
            outer_theta = 0.05
            B = 0.1
            parameters0 = parameters[0]
            for t in range(args.T):
                parameters_y = parameters[0] + (1 - inner_theta) * (parameters[0] - parameters0)
                loss_train = train_loss([parameters_y], hparams, [x_train, y_train])
                inner_grad = torch.autograd.grad(loss_train, [parameters_y])
                parameters0, parameters[0] = parameters[0], parameters_y - args.inner_lr * inner_grad[0]
                inner_losses.append(loss_train)

                if t % train_log_interval == 0 or t == args.T - 1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))

            hparams_y = hparams[0] + (1 - outer_theta) * (hparams[0] - hparams0)
            hg.CG(parameters, [hparams_y], args.hessian_q, inner_opt_cg, val_loss, stochastic=False, tol=tol)
            hparams0 = hparams[0]
            hparams[0] = hparams_y - args.outer_lr * hparams_y.grad
            s += float(torch.norm(hparams[0] - hparams0)) ** 2
            k += 1
            if k * s > B ** 2:
                hparams[0] = hparams0 + torch.rand_like(hparams0) * r
                k, s = 0, 0

            calls_num += args.batch_size * (args.hessian_q + args.T)
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

        elif args.alg == 'RAHGD':
            inner_losses = []
            inner_theta = 0.009  # parameter of AGD
            outer_theta = 0.05
            B = 0.1
            parameters0 = parameters[0]
            for t in range(args.T):
                parameters_y = parameters[0] + (1 - inner_theta) * (parameters[0] - parameters0)
                loss_train = train_loss([parameters_y], hparams, [x_train, y_train])
                inner_grad = torch.autograd.grad(loss_train, [parameters_y])
                parameters0, parameters[0] = parameters[0], parameters_y - args.inner_lr * inner_grad[0]
                inner_losses.append(loss_train)

                if t % train_log_interval == 0 or t == args.T - 1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))

            hparams_y = hparams[0] + (1 - outer_theta) * (hparams[0] - hparams0)
            hg.CG(parameters, [hparams_y], args.hessian_q, inner_opt_cg, val_loss, stochastic=False, tol=tol)
            hparams0 = hparams[0]
            hparams[0] = hparams_y - args.outer_lr * hparams_y.grad
            s += float(torch.norm(hparams[0] - hparams0)) ** 2
            k += 1
            if k * s > B ** 2:
                hparams[0] = hparams0
                k, s = 0, 0

            calls_num += args.batch_size * (args.hessian_q + args.T)
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

        elif args.alg == 'ITD':
            inner_losses = []
            for t in range(args.T):
                loss_train = train_loss(parameters, hparams, [x_train, y_train])
                inner_grad = torch.autograd.grad(loss_train, parameters, create_graph=True)
                parameters[0] = parameters[0] - args.inner_lr * inner_grad[0]
                inner_losses.append(loss_train)

                if t % train_log_interval == 0 or t == args.T - 1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))
            loss_val = val_loss(parameters, hparams[0])
            outer_grad = torch.autograd.grad(loss_val, hparams[0])[0]
            hparams[0] = hparams[0] - args.outer_lr * outer_grad
            calls_num += args.batch_size * (1 + args.T)
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

        elif args.alg == 'AID' or args.alg == 'PAID' or args.alg == 'BA-CG':
            if args.alg == 'AID':
                r = 0
                iteration = args.T
            elif args.alg == 'PAID':
                r = 0.1
                iteration = args.T
            else:
                r = 0
                iteration = int(math.pow(o_step+1, 1/4)*2)+1
            inner_losses = []
            for t in range(iteration):
                loss_train = train_loss(parameters, hparams, [x_train, y_train])
                inner_grad = torch.autograd.grad(loss_train, parameters)
                parameters[0] = parameters[0] - args.inner_lr * inner_grad[0]
                inner_losses.append(loss_train)

                if t % train_log_interval == 0 or t == iteration - 1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))

            hg.CG(parameters, hparams, args.hessian_q, inner_opt_cg, val_loss, stochastic=False, tol=tol)
            if torch.norm(hparams[0].grad) <= 0.8 * tol * 1e8 and o_step - pk >= huaT:
                hparams[0] = hparams[0] - args.outer_lr * (torch.rand_like(hparams[0]) * r + hparams[0].grad)
                pk = o_step
            else:
                hparams[0] = hparams[0] - args.outer_lr * hparams[0].grad

            calls_num += args.batch_size * (args.hessian_q + iteration)
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

        elif args.alg == 'IFSBA':
            # IFSBA: Lower-level First-order Second-order Bilevel Algorithm
            # Inner loop: AGD for parameters
            # Outer loop: Cubic Regularization with Lazy Hessian for hyperparameters
            
            # Inner optimization: AGD for parameters
            inner_losses = []
            parameters0 = parameters[0].clone().detach()
            
            for t in range(args.T):
                # AGD update: y = x + (1-theta) * (x - x0)
                parameters_y = parameters[0] + (1 - args.theta1) * (parameters[0] - parameters0)
                parameters_y.requires_grad_(True)
                
                # Compute gradient
                loss_train = train_loss([parameters_y], hparams, [x_train, y_train])
                inner_grad = torch.autograd.grad(loss_train, [parameters_y], create_graph=False)[0]
                
                # Update: x_new = y - lr * grad
                parameters0 = parameters[0].clone().detach()
                parameters[0] = (parameters_y - args.inner_lr * inner_grad).detach()
                parameters[0].requires_grad_(True)
                
                inner_losses.append(loss_train)
                
                if t % train_log_interval == 0 or t == args.T - 1:
                    print('t={} loss: {}'.format(t, inner_losses[-1]))
            
            # Outer optimization: Cubic Regularization with Lazy Hessian
            is_hessian_update_epoch = (o_step % args.m == 0)
            
            # Compute hyperparameter gradient using implicit differentiation (similar to AID)
            outer_loss = lambda x, w: val_loss(x, w)
            inner_loss = lambda x, w, d: train_loss(x, w, d)
            inner_opt_cg = hg.GradientDescent(inner_loss, 1., data_or_iter=train_iterator)
            
            # Use CG to compute hyperparameter gradient
            hparams[0].requires_grad_(True)
            hg.CG([parameters[0]], hparams, args.hessian_q, inner_opt_cg, outer_loss, 
                  stochastic=False, set_grad=True)
            hp_grad = hparams[0].grad.clone()
            
            # Update Hessian function if needed
            if is_hessian_update_epoch:
                print(f"[IFSBA] Epoch {o_step}: Updating Hessian (Lazy Hessian)")
                
                # Capture snapshots for lazy Hessian
                parameters_snapshot = [p.detach().clone().requires_grad_(True) for p in parameters]
                hparams_snapshot = [hp.detach().clone().requires_grad_(True) for hp in hparams]
                
                def w_hessian_vector_product(v):
                    """Compute Hessian-vector product H @ v for hyperparameters using Chebyshev"""
                    # Re-establish computation graph by running inner optimization
                    # This creates dependency between hparams and the resulting parameters
                    params_dep = [p.detach().clone().requires_grad_(True) for p in parameters_snapshot]
                    hparams_dep = [hp.detach().clone().requires_grad_(True) for hp in hparams_snapshot]
                    
                    # Run one inner optimization step to establish dependency
                    # This makes params_dep depend on hparams_dep through train_loss
                    loss_inner = train_loss(params_dep, hparams_dep, [x_train, y_train])
                    grad_inner = torch.autograd.grad(
                        loss_inner, 
                        params_dep, 
                        create_graph=True,
                        retain_graph=True
                    )[0]
                    
                    # Update parameters - this creates dependency on hparams
                    params_updated = [params_dep[0] - 0.001 * grad_inner]
                    if len(params_dep) > 1:
                        params_updated.extend(params_dep[1:])
                    params_updated = [p.requires_grad_(True) for p in params_updated]
                    
                    # Now compute validation loss - it depends on hparams through params_updated
                    loss_val = val_loss(params_updated, hparams_dep)
                    
                    # Compute gradient w.r.t. hyperparameters
                    # Now hparams_dep is in the computation graph
                    gw = torch.autograd.grad(
                        loss_val,
                        hparams_dep,
                        create_graph=True,
                        retain_graph=True,
                        allow_unused=True
                    )[0]
                    
                    if gw is None:
                        return torch.zeros_like(v)
                    
                    # Compute H @ v = d/dhparams (gw^T @ v) using Chebyshev
                    # This is the Hessian-vector product
                    Hv = torch.autograd.grad(
                        outputs=gw,
                        inputs=hparams_dep,
                        grad_outputs=v,
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True
                    )[0]
                    
                    if Hv is None:
                        Hv = torch.zeros_like(v)
                    
                    return Hv.detach()
                
                cached_hessian_func = w_hessian_vector_product
            
            # Update hyperparameters using cubic regularization
            if args.use_cubic:
                if cached_hessian_func is None:
                    # First iteration: use identity Hessian
                    print(f"[IFSBA] Epoch {o_step}: Using identity Hessian (first iteration)")
                    cached_hessian_func = lambda v: v.clone()
                
                # Use cubic_newton_step_chebyshev at EVERY iteration
                # The difference is that cached_hessian_func uses old snapshots in lazy epochs
                hp_step = cubic_newton_step_chebyshev(
                    grad=hp_grad,
                    hessian_func=cached_hessian_func,  # Use cached Hessian in lazy epochs
                    M=args.M,
                    K=args.cheb_K,
                    l_est=args.l_est,
                    mu_est=args.mu_est,
                    max_iters=args.cubic_iters,
                    tol=1e-4
                )
                
                with torch.no_grad():
                    hparams[0].data = hparams[0].data + hp_step
            else:
                # Fallback: first-order gradient descent
                with torch.no_grad():
                    hparams[0].data = hparams[0].data - args.outer_lr * hp_grad
            
            calls_num += args.batch_size * (args.hessian_q + args.T + args.cubic_iters * args.cheb_K * 3 )
            final_params = parameters
            for p, new_p in zip(parameters, final_params[:len(parameters)]):
                if warm_start:
                    p.data = new_p
                else:
                    p.data = torch.zeros_like(p)
            val_loss(final_params, hparams)

        elif args.alg == 'F2BA':
            # ======================================================
            # F2BA: First-order First-order Bilevel Algorithm
            # g: inner loss = train_loss
            # f: outer loss = val_loss
            # ======================================================

            tau = args.f2ba_tau
            alpha = args.f2ba_alpha
            lmbd = args.lmbd
            K = args.f2ba_K

            # Initialize y_t, z_t
            y_t = parameters[0].detach().clone().requires_grad_(True)
            z_t = parameters[0].detach().clone().requires_grad_(True)

            # -------------------------------
            # Inner loop: k = 0,...,K-1
            # -------------------------------
            for k in range(K):

                # ----- z update: z^{k+1} = z^k - alpha * ∇_y g(x_t, z^k)
                loss_g_z = train_loss([z_t], hparams, [x_train, y_train])
                grad_z = torch.autograd.grad(loss_g_z, z_t)[0]
                z_t = (z_t - alpha * grad_z).detach().requires_grad_(True)

                # ----- y update:
                # y^{k+1} = y^k - tau ( ∇_y f + lambda ∇_y g )
                loss_f_y = val_loss([y_t], hparams)
                loss_g_y = train_loss([y_t], hparams, [x_train, y_train])

                grad_f = torch.autograd.grad(loss_f_y, y_t, retain_graph=True)[0]
                grad_g = torch.autograd.grad(loss_g_y, y_t)[0]

                y_t = (y_t - tau * (grad_f + lmbd * grad_g)).detach().requires_grad_(True)

            # ---------------------------------------------------
            # Approximate hypergradient:
            # ∇_x f(x, y^K) + λ ( ∇_x g(x, y^K) - ∇_x g(x, z^K) )
            # ---------------------------------------------------
            loss_f_yK = val_loss([y_t], hparams)
            loss_g_yK = train_loss([y_t], hparams, [x_train, y_train])
            loss_g_zK = train_loss([z_t], hparams, [x_train, y_train])

            grad_fx = torch.autograd.grad(loss_f_yK, hparams, retain_graph=True, allow_unused=True)[0]
            grad_gy = torch.autograd.grad(loss_g_yK, hparams, retain_graph=True, allow_unused=True)[0]
            grad_gz = torch.autograd.grad(loss_g_zK, hparams, allow_unused=True)[0]

            # Handle None gradients (val_loss doesn't depend on hparams)
            if grad_fx is None:
                grad_fx = torch.zeros_like(hparams[0])
            if grad_gy is None:
                grad_gy = torch.zeros_like(hparams[0])
            if grad_gz is None:
                grad_gz = torch.zeros_like(hparams[0])

            hyper_grad = grad_fx + lmbd * (grad_gy - grad_gz)

            # -------------------------------
            # Outer update: x_{t+1} = x_t - eta * hyper_grad
            # -------------------------------
            with torch.no_grad():
                hparams[0].data -= args.outer_lr * hyper_grad

            # warm start inner params
            parameters[0].data = y_t.detach()

            # accounting
            calls_num += args.batch_size * (2 * K + 3)
            val_loss(parameters, hparams)

        iter_time = time.time() - start_time
        total_time += iter_time
        if o_step % val_log_interval == 0 or o_step == args.epochs-1:
            test_loss, test_acc = eval(parameters, x_test, y_test)
            loss_acc_time_results[o_step+1, 0] = test_loss
            loss_acc_time_results[o_step+1, 1] = test_acc
            loss_acc_time_results[o_step+1, 2] = total_time
            loss_acc_time_results[o_step+1, 3] = calls_num
            print('o_step={} ({:.2e}s) Val loss: {:.4e}, Val Acc: {:.2f}%'.format(o_step, iter_time, val_losses[-1],
                                                                                100*val_accs[-1]))
            print('          Test loss: {:.4e}, Test Acc: {:.2f}%'.format(test_loss, 100*test_acc))

    file_name = 'results.npy'
    file_addr = os.path.join(args.save_folder, file_name)
    with open(file_addr, 'wb') as f:
            np.save(f, loss_acc_time_results)

    print(loss_acc_time_results)
    print('HPO ended in {:.2e} seconds\n'.format(total_time))


# ============================================
# IFSBA Helper Functions
# ============================================

def hessian_vector_product_chebyshev(v, hessian_func, K, l_est, mu_est):
    """
    Approximate H @ v using Chebyshev polynomial, where H is the Hessian.
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
    """
    Solve cubic regularized subproblem using Chebyshev approximation.
    """
    device = grad.device
    gnorm = grad.norm()

    # If gradient tiny → no movement
    if gnorm < 1e-6:
        return torch.zeros_like(grad)

    # ============================================================
    # 1. Analyzing the Cauchy cubic step (large-gradient regime)
    # ============================================================
    if gnorm >= 0.01:   # The threshold is tunable; typically values in the range 1e-3～1e-2 work well.

        # Compute Hg using Chebyshev approx
        Hg = hessian_vector_product_chebyshev(
            grad, hessian_func, K, l_est, mu_est
        )

        gHg = (grad * Hg).sum()           # scalar g^T H g

        # ---- Closed-form expression of Rc ----
        # Rc = - (gHg / (M ||g||^2)) + sqrt( (gHg/(M||g||^2))^2 + 2||g||/M )
        alpha = gHg / (M * (gnorm ** 2) + 1e-12)
        Rc = -alpha + torch.sqrt(alpha * alpha + 2 * gnorm / M)

        # Cauchy step
        s = -Rc * grad / (gnorm + 1e-12)
        return s

    # ============================================================
    # 2. Small-gradient regime: Iterative cubic solver
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


# choices=['PRAHGD', 'RAHGD', 'BA-CG', 'AID', 'ITD', 'PAID', 'IFSBA', 'F2BA']
def main():
    args = parse_args()
    # Note: Remove the hardcoded line below to use command-line argument
    # args.alg = 'ITD'  # Uncomment this line to force a specific algorithm
    print(f"Running algorithm: {args.alg}")
    print(f"Arguments: {args}")
    train_model(args)


if __name__ == '__main__':
    main()
