#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
IFSBA-style few-shot meta-learning with adaptation over partial parameters.

Main fixes:
1. Do NOT update hparams before all old-graph HVP uses are finished.
2. Remove unsafe .data assignment patterns.
3. Make evaluate() compatible with current Task implementation.
4. Add T_test argument.
"""

import math
import argparse
import time
import os
import copy
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as Tr

import higher
import learn2learn as l2l
from learn2learn.data.transforms import (
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)

from networks import OmniglotNetFeats, MiniimageNetFeats, classifier
from utils import Lambda, load_checkpoint, save_checkpoint


# ============================================================
# Helpers
# ============================================================

def split_into_adapt_eval(batch, shots, ways, device=None):
    """
    Splits task data into adaptation/support set and evaluation/query set.
    Assumes 2 * shots samples per class are sampled.
    """
    data, labels = batch
    data, labels = data.to(device), labels.to(device)

    adapt_idx = np.zeros(data.size(0), dtype=bool)
    adapt_idx[np.arange(shots * ways) * 2] = True

    eval_idx = torch.from_numpy(~adapt_idx)
    adapt_idx = torch.from_numpy(adapt_idx)

    adapt_data, adapt_labels = data[adapt_idx], labels[adapt_idx]
    eval_data, eval_labels = data[eval_idx], labels[eval_idx]

    return adapt_data, adapt_labels, eval_data, eval_labels


def clone_params(params, requires_grad=True):
    return [p.detach().clone().requires_grad_(requires_grad) for p in params]


def detach_params(params, requires_grad=False):
    return [p.detach().clone().requires_grad_(requires_grad) for p in params]


def safe_grads(loss, params, create_graph=False, retain_graph=False):
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    out = []
    for p, g in zip(params, grads):
        out.append(torch.zeros_like(p) if g is None else g)
    return out


def update_tensor_grads(params, grads):
    for p, g in zip(params, grads):
        if g is None:
            continue
        if p.grad is None:
            p.grad = g.detach().clone()
        else:
            p.grad = p.grad + g.detach()


def flatten_tensors(tensors):
    return torch.cat([t.reshape(-1) for t in tensors])


def unflatten_like(vec, like_tensors):
    outs = []
    idx = 0
    for t in like_tensors:
        numel = t.numel()
        outs.append(vec[idx:idx + numel].view_as(t))
        idx += numel
    return outs


# ============================================================
# Task object
# ============================================================

class Task:
    """
    One few-shot task.

    x := hparams (meta/backbone params)
    y,w := task-specific classifier params

    g(x; ·) := support/adaptation loss
    f(x; ·) := query/evaluation loss
    """

    def __init__(self, reg_param, meta_model, task_model, data, batch_size=None):
        device = next(meta_model.parameters()).device

        self.fmeta = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)
        self.ftask = higher.monkeypatch(task_model, device=device, copy_initial_weights=True)

        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def reg_f(self, params):
        return sum((p ** 2).sum() for p in params)

    def g_loss(self, hparams, params, divide_by_batch=False):
        """
        Lower-level objective: support loss.
        """
        feats = self.fmeta(self.train_input, params=hparams)
        out = self.ftask(feats, params=params)
        loss = F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.reg_f(params)
        if divide_by_batch:
            loss = loss / self.batch_size
        return loss

    def f_loss(self, hparams, params, divide_by_batch=False, record=False):
        """
        Upper-level objective: query loss.
        """
        feats = self.fmeta(self.test_input, params=hparams)
        out = self.ftask(feats, params=params)
        loss = F.cross_entropy(out, self.test_target)

        if divide_by_batch:
            loss = loss / self.batch_size

        if record:
            self.val_loss = loss.item()
            pred = out.argmax(dim=1, keepdim=True)
            self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return loss

    def val_loss_f(self, params, hparams):
        return self.f_loss(hparams, params, divide_by_batch=True, record=True)


# ============================================================
# AGD subroutines
# ============================================================

def AGD_lower(task, hparams, w_init, K1, lr1, theta1, log_interval=None):
    """
    AGD on g(x; ·)
    """
    w = clone_params(w_init, requires_grad=False)
    v = clone_params(w_init, requires_grad=True)

    for k in range(K1):
        g_val = task.g_loss(hparams, v, divide_by_batch=False)
        grad = safe_grads(g_val, v, create_graph=False, retain_graph=False)

        with torch.no_grad():
            w_new = [vi - lr1 * gi for vi, gi in zip(v, grad)]
            v_new = [wn + theta1 * (wn - w_old) for wn, w_old in zip(w_new, w)]

        w = [p.detach() for p in w_new]
        v = [p.detach().requires_grad_(True) for p in v_new]

        if log_interval and (k % log_interval == 0 or k == K1 - 1):
            print(f"[AGD_lower] k={k}, g={g_val.item():.6f}")

    return detach_params(w, requires_grad=False)


def AGD_upper_inner(task, hparams, y_init, lam, K2, lr2, theta2, log_interval=None):
    """
    AGD on L_lambda(x; y) = f(x,y) + lambda * g(x,y)
    """
    y = clone_params(y_init, requires_grad=False)
    v = clone_params(y_init, requires_grad=True)

    for k in range(K2):
        f_val = task.f_loss(hparams, v, divide_by_batch=False, record=False)
        g_val = task.g_loss(hparams, v, divide_by_batch=False)
        L_val = f_val + lam * g_val
        grad = safe_grads(L_val, v, create_graph=False, retain_graph=False)

        with torch.no_grad():
            y_new = [vi - lr2 * gi for vi, gi in zip(v, grad)]
            v_new = [yn + theta2 * (yn - y_old) for yn, y_old in zip(y_new, y)]

        y = [p.detach() for p in y_new]
        v = [p.detach().requires_grad_(True) for p in v_new]

        if log_interval and (k % log_interval == 0 or k == K2 - 1):
            print(f"[AGD_upper_inner] k={k}, L={L_val.item():.6f}")

    return detach_params(y, requires_grad=False)


# ============================================================
# Outer surrogate gradient / HVP
# ============================================================

def outer_surrogate_grad(task, hparams, y_t, w_t, lam, divide_by_batch=True, record=True):
    """
    g_t = ∇_x f(x; y_t) + λ ( ∇_x g(x; y_t) - ∇_x g(x; w_t) )
    """
    f_val = task.f_loss(hparams, y_t, divide_by_batch=divide_by_batch, record=record)
    g_y_val = task.g_loss(hparams, y_t, divide_by_batch=divide_by_batch)
    g_w_val = task.g_loss(hparams, w_t, divide_by_batch=divide_by_batch)

    grad_f_x = safe_grads(f_val, hparams, create_graph=True, retain_graph=True)
    grad_g_y_x = safe_grads(g_y_val, hparams, create_graph=True, retain_graph=True)
    grad_g_w_x = safe_grads(g_w_val, hparams, create_graph=True, retain_graph=True)

    g_t = [
        gf + lam * (ggy - ggw)
        for gf, ggy, ggw in zip(grad_f_x, grad_g_y_x, grad_g_w_x)
    ]
    return g_t


def make_outer_hvp(task, hparams, y_t, w_t, lam, divide_by_batch=True):
    """
    Returns:
        g_t_list : list of tensors, outer surrogate gradient wrt hparams
        hvp_func : callable(v_flat) -> H @ v_flat
    """
    g_t = outer_surrogate_grad(
        task=task,
        hparams=hparams,
        y_t=y_t,
        w_t=w_t,
        lam=lam,
        divide_by_batch=divide_by_batch,
        record=False,
    )
    g_flat = flatten_tensors(g_t)

    def hvp_func(v_flat):
        hv = torch.autograd.grad(
            g_flat,
            hparams,
            grad_outputs=v_flat,
            retain_graph=True,
            allow_unused=True,
        )
        out = []
        for p, h in zip(hparams, hv):
            out.append(torch.zeros_like(p) if h is None else h)
        return flatten_tensors(out)

    return g_t, hvp_func


# ============================================================
# Chebyshev HVP + cubic solver
# ============================================================

def hessian_vector_product_chebyshev(v, hessian_func, K, l_est, mu_est):
    """
    Approximate H @ v using Chebyshev polynomial,
    where hessian_func(u) returns exact HVP H @ u.
    v is flattened.
    """
    mu1 = mu_est / (2.0 * l_est)
    l1 = 0.5

    if l_est <= 0 or mu_est <= 0:
        raise ValueError(f"Require l_est>0 and mu_est>0, got l_est={l_est}, mu_est={mu_est}.")
    if mu1 >= l1:
        # keep it numerically valid
        mu1 = 0.99 * l1

    p1 = 2.0 / (l1 - mu1)
    p2 = (l1 + mu1) / (l1 - mu1)
    p3 = (math.sqrt(mu1 / l1) - 1.0) / (math.sqrt(mu1 / l1) + 1.0)
    c = 2.0 / math.sqrt(l1 * mu1)

    T_prev = v.clone()
    Hv = hessian_func(v) / (2.0 * l_est)
    T_curr = p1 * Hv - p2 * v

    result = c / 2.0 * T_prev
    c = c * p3
    result = result + c * T_curr

    for k in range(2, K):
        c = c * p3
        Hv = hessian_func(T_curr) / (2.0 * l_est)
        T_next = 2.0 * p1 * Hv - 2.0 * p2 * T_curr - T_prev
        result = result + c * T_next
        T_prev = T_curr
        T_curr = T_next

    result = result / (2.0 * l_est)
    return result


def cubic_newton_step_chebyshev(
    grad_flat,
    hessian_func,
    sigma,
    K_cheby,
    l_est,
    mu_est,
    max_iters=10,
    tol=1e-4,
):
    """
    Approximate cubic-regularized Newton step:
        min_s <g, s> + 1/2 s^T H s + sigma/6 ||s||^3
    """
    gnorm = grad_flat.norm()

    if gnorm < 1e-8:
        return torch.zeros_like(grad_flat)

    # Large-gradient regime: Cauchy-like step
    if gnorm >= 1e-2:
        Hg = hessian_vector_product_chebyshev(
            grad_flat, hessian_func, K_cheby, l_est, mu_est
        )
        gHg = (grad_flat * Hg).sum()
        alpha = gHg / (sigma * (gnorm ** 2) + 1e-12)
        Rc = -alpha + torch.sqrt(alpha * alpha + 2.0 * gnorm / sigma)
        s = -Rc * grad_flat / (gnorm + 1e-12)
        return s

    # Small-gradient regime: iterative solver
    s = torch.zeros_like(grad_flat)
    lr = 0.1

    for _ in range(max_iters):
        Hs = hessian_vector_product_chebyshev(
            s, hessian_func, K_cheby, l_est, mu_est
        )

        s_norm = s.norm()
        if s_norm < 1e-10:
            cubic_grad = grad_flat + Hs
        else:
            cubic_grad = grad_flat + Hs + (sigma / 2.0) * s_norm * s

        if cubic_grad.norm() < tol:
            break

        s = s - lr * cubic_grad

    return s


def cubic_model_value(grad_flat, hessian_func, s_flat, sigma):
    """
    m(s) = <g,s> + 1/2 s^T H s + sigma/6 ||s||^3
    """
    Hs = hessian_func(s_flat)
    return (
        (grad_flat * s_flat).sum()
        + 0.5 * (s_flat * Hs).sum()
        + (sigma / 6.0) * (s_flat.norm() ** 3)
    )


# ============================================================
# One IFSBA-style outer step
# ============================================================

def train_one_outer_step_ifsba(
    train_tasks,
    meta_model,
    task_model,
    hparams,
    y0,
    w0,
    reg_param,
    shots,
    ways,
    meta_bsz,
    lam,
    K1,
    K2,
    lr1,
    lr2,
    theta1,
    theta2,
    sigma,
    M_bound,
    cheby_K,
    l_est,
    mu_est,
    eps,
    device,
    inner_log_interval=None,
):
    """
    One outer iteration of an IFSBA-style meta-learning algorithm.
    """
    total_loss = 0.0
    total_acc = 0.0

    y_accum = [torch.zeros_like(p).to(device) for p in y0]
    w_accum = [torch.zeros_like(p).to(device) for p in w0]

    grad_accum_flat = None
    hvp_cache = []

    for _ in range(meta_bsz):
        task_data = train_tasks.sample()
        task_data = split_into_adapt_eval(
            task_data,
            shots=shots,
            ways=ways,
            device=device,
        )
        task = Task(reg_param, meta_model, task_model, task_data, batch_size=meta_bsz)

        w_t = AGD_lower(
            task=task,
            hparams=hparams,
            w_init=w0,
            K1=K1,
            lr1=lr1,
            theta1=theta1,
            log_interval=inner_log_interval,
        )

        y_t = AGD_upper_inner(
            task=task,
            hparams=hparams,
            y_init=y0,
            lam=lam,
            K2=K2,
            lr2=lr2,
            theta2=theta2,
            log_interval=inner_log_interval,
        )

        g_t_list, hvp_func = make_outer_hvp(
            task=task,
            hparams=hparams,
            y_t=y_t,
            w_t=w_t,
            lam=lam,
            divide_by_batch=True,
        )
        g_t_flat = flatten_tensors(g_t_list)

        grad_accum_flat = g_t_flat if grad_accum_flat is None else grad_accum_flat + g_t_flat
        hvp_cache.append(hvp_func)

        task.val_loss_f(y_t, hparams)
        total_loss += task.val_loss
        total_acc += task.val_acc / meta_bsz

        y_accum = [a + y / meta_bsz for a, y in zip(y_accum, y_t)]
        w_accum = [a + w / meta_bsz for a, w in zip(w_accum, w_t)]

    grad_flat = grad_accum_flat / meta_bsz

    def avg_hvp(v_flat):
        out = None
        for hvp in hvp_cache:
            hv = hvp(v_flat)
            out = hv if out is None else out + hv
        return out / meta_bsz

    # First cubic step on CURRENT graph / CURRENT hparams
    s_flat = cubic_newton_step_chebyshev(
        grad_flat=grad_flat,
        hessian_func=avg_hvp,
        sigma=sigma,
        K_cheby=cheby_K,
        l_est=l_est,
        mu_est=mu_est,
        max_iters=10,
        tol=1e-4,
    )

    delta_t = cubic_model_value(grad_flat, avg_hvp, s_flat, sigma)

    # IMPORTANT FIX:
    # Do NOT update hparams before finishing all uses of avg_hvp / old graph.
    final_step_flat = s_flat

    if delta_t > -(eps ** 3) / (128.0 * M_bound):
        final_step_flat = cubic_newton_step_chebyshev(
            grad_flat=grad_flat,
            hessian_func=avg_hvp,
            sigma=sigma,
            K_cheby=cheby_K,
            l_est=l_est,
            mu_est=mu_est,
            max_iters=30,
            tol=min(1e-6, eps * 0.1),
        )

    # Now update hparams ONCE, after all old-graph HVP computations are done
    final_step_struct = unflatten_like(final_step_flat, hparams)
    with torch.no_grad():
        for hp, s in zip(hparams, final_step_struct):
            hp.add_(s)

    new_y0 = [p.detach().clone() for p in y_accum]
    new_w0 = [p.detach().clone() for p in w_accum]

    return new_y0, new_w0, total_loss, total_acc, delta_t.item()


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    metadataset,
    meta_model,
    task_model,
    hparams,
    w0,
    reg_param,
    inner_lr,
    inner_mu,
    inner_steps,
    shots,
    ways,
):
    device = next(meta_model.parameters()).device

    iters = metadataset.num_tasks
    eval_losses, eval_accs = [], []

    for k in range(iters):
        data = metadataset.sample()
        data = split_into_adapt_eval(
            data,
            shots=shots,
            ways=ways,
            device=device,
        )

        task = Task(reg_param, meta_model, task_model, data)

        params = [p.detach().clone().requires_grad_(True) for p in w0]
        velocity = [torch.zeros_like(p) for p in params]

        for _ in range(inner_steps):
            loss = task.g_loss(hparams, params, divide_by_batch=False)
            grads = torch.autograd.grad(loss, params, allow_unused=True)

            new_params = []
            new_velocity = []
            for p, v, g in zip(params, velocity, grads):
                if g is None:
                    g = torch.zeros_like(p)
                v_new = inner_mu * v - inner_lr * g
                p_new = (p + v_new).detach().requires_grad_(True)
                new_params.append(p_new)
                new_velocity.append(v_new.detach())

            params = new_params
            velocity = new_velocity

        task.val_loss_f(params, hparams)
        eval_losses.append(task.val_loss)
        eval_accs.append(task.val_acc)

        if k >= 999:
            return np.array(eval_losses), np.array(eval_accs)

    return np.array(eval_losses), np.array(eval_accs)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='IFSBA-style Meta-Learning with Partial Parameter Adaptation')

    # General
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='fc100',
                        help='omniglot or miniimagenet or fc100')
    parser.add_argument('--run_id', type=int, default=1, help='which run this is, e.g. 1,2,3,4,5')
    parser.add_argument('--resume', type=bool, default=False)
    parser.add_argument('--ckpt_dir', type=str, default='metalogs')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--use_resnet', type=bool, default=False)
    parser.add_argument('--no-cuda', action='store_true', default=False)

    # IFSBA hyperparameters
    parser.add_argument('--lmbd', type=float, default=1.0)

    parser.add_argument('--K1', type=int, default=20, help='AGD steps for lower problem')
    parser.add_argument('--K2', type=int, default=20, help='AGD steps for upper-inner problem')
    parser.add_argument('--T_test', type=int, default=20, help='inner steps for evaluation')

    parser.add_argument('--l1', type=float, default=100.0, help='smoothness estimate for lower AGD')
    parser.add_argument('--l2', type=float, default=100.0, help='smoothness estimate for upper-inner AGD')
    parser.add_argument('--kappa1', type=float, default=10.0, help='condition number estimate for lower AGD')
    parser.add_argument('--kappa2', type=float, default=10.0, help='condition number estimate for upper-inner AGD')

    parser.add_argument('--sigma', type=float, default=1.0, help='cubic regularization coefficient')
    parser.add_argument('--M', type=float, default=1.0, help='Hessian Lipschitz bound used in stopping criterion')
    parser.add_argument('--eps', type=float, default=1e-3, help='target stationarity tolerance')

    parser.add_argument('--cheby_K', type=int, default=10)
    parser.add_argument('--l_est', type=float, default=10.0)
    parser.add_argument('--mu_est', type=float, default=1.0)

    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    run = args.run_id
    seed = args.seed if args.seed is not None else run
    inner_lr = 0.01
    inner_mu = 0.9
    K_outer = args.steps
    T_test = args.T_test

    stop_k = None
    n_tasks_train = 20000
    n_tasks_val = 200
    n_tasks_test = 200
    log_interval = 25
    eval_interval = 50

    lr1 = 1.0 / args.l1
    lr2 = 1.0 / args.l2
    theta1 = (math.sqrt(args.kappa1) - 1.0) / (math.sqrt(args.kappa1) + 1.0)
    theta2 = (math.sqrt(args.kappa2) - 1.0) / (math.sqrt(args.kappa2) + 1.0)

    if args.dataset == 'omniglot':
        reg_param = 0.2
    elif args.dataset in ['miniimagenet', 'fc100']:
        reg_param = 0.5
    else:
        raise NotImplementedError(args.dataset)

    loc = locals().copy()
    del loc['parser']
    del loc['args']

    log_path = os.path.join(args.ckpt_dir, f'log_IFSBA_{args.dataset}_run{run}.txt')
    out_file = open(log_path, 'w')

    string = f"+++++++++++++++++++ Run {run} Arguments ++++++++++++++++++++\n"
    string += f"effective_seed:{seed}\n"
    for item, value in args.__dict__.items():
        string += f"{item}:{value}\n"
    out_file.write(string + '\n')
    out_file.flush()
    print(string)

    string = f"+++++++++++++++++++ Run {run} Local Variables ++++++++++++++++++++\n"
    for item, value in loc.items():
        string += f"{item}:{value}\n"
    out_file.write(string + '\n')
    out_file.flush()
    print(string)

    cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    print('Training on cuda device...' if cuda else 'Training on cpu...')

    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Optional: use this when debugging autograd errors
    # torch.autograd.set_detect_anomaly(True)

    # --------------------------------------------------------
    # Dataset + model
    # --------------------------------------------------------
    if args.dataset == 'omniglot':
        train_tasks, val_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets(
            'omniglot',
            train_ways=args.ways,
            train_samples=2 * args.shots,
            test_ways=args.ways,
            test_samples=2 * args.shots,
            num_tasks=10000,
            root='data/omniglot'
        )
        meta_model = OmniglotNetFeats(64).to(device)
        task_model = classifier(64, args.ways).to(device)

    elif args.dataset == 'miniimagenet':
        MEAN = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        STD = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = Tr.Normalize(mean=MEAN, std=STD)

        transform_train = Tr.Compose([normalize])
        transform_test = Tr.Compose([normalize])

        train_dataset = l2l.vision.datasets.MiniImagenet(
            root='data/MiniImageNet',
            mode='train',
            transform=transform_train,
            download=False
        )
        val_dataset = l2l.vision.datasets.MiniImagenet(
            root='data/MiniImageNet',
            mode='validation',
            transform=transform_test,
            download=False
        )
        test_dataset = l2l.vision.datasets.MiniImagenet(
            root='data/MiniImageNet',
            mode='test',
            transform=transform_test,
            download=False
        )

        if args.use_resnet:
            raise NotImplementedError("ResNet12 branch not filled here. Use MiniimageNetFeats or add your ResNet12.")
        else:
            meta_model = MiniimageNetFeats(32).to(device)
            task_model = classifier(32 * 5 * 5, args.ways).to(device)

        train_dataset = l2l.data.MetaDataset(train_dataset)
        val_dataset = l2l.data.MetaDataset(val_dataset)
        test_dataset = l2l.data.MetaDataset(test_dataset)

        train_transforms = [
            FusedNWaysKShots(train_dataset, n=args.ways, k=2 * args.shots),
            LoadData(train_dataset),
            RemapLabels(train_dataset),
            ConsecutiveLabels(train_dataset),
        ]
        val_transforms = [
            FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
            LoadData(val_dataset),
            ConsecutiveLabels(val_dataset),
            RemapLabels(val_dataset),
        ]
        test_transforms = [
            FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
            LoadData(test_dataset),
            RemapLabels(test_dataset),
            ConsecutiveLabels(test_dataset),
        ]

        train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=n_tasks_train)
        val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms=val_transforms, num_tasks=n_tasks_val)
        test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=n_tasks_test)

    elif args.dataset == 'fc100':
        train_dataset = l2l.vision.datasets.FC100(
            root='data/FC100',
            transform=Tr.ToTensor(),
            mode='train',
            download=True
        )
        val_dataset = l2l.vision.datasets.FC100(
            root='data/FC100',
            transform=Tr.ToTensor(),
            mode='validation',
            download=True
        )
        test_dataset = l2l.vision.datasets.FC100(
            root='data/FC100',
            transform=Tr.ToTensor(),
            mode='test',
            download=True
        )

        train_dataset = l2l.data.MetaDataset(train_dataset)
        val_dataset = l2l.data.MetaDataset(val_dataset)
        test_dataset = l2l.data.MetaDataset(test_dataset)

        train_transforms = [
            FusedNWaysKShots(train_dataset, n=args.ways, k=2 * args.shots),
            LoadData(train_dataset),
            RemapLabels(train_dataset),
            ConsecutiveLabels(train_dataset),
        ]
        val_transforms = [
            FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
            LoadData(val_dataset),
            ConsecutiveLabels(val_dataset),
            RemapLabels(val_dataset),
        ]
        test_transforms = [
            FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
            LoadData(test_dataset),
            RemapLabels(test_dataset),
            ConsecutiveLabels(test_dataset),
        ]

        train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=n_tasks_train)
        val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms=val_transforms, num_tasks=n_tasks_val)
        test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=n_tasks_test)

        meta_model = torch.nn.Sequential(
            l2l.vision.models.ConvBase(hidden=64, channels=3, max_pool=True),
            Lambda(lambda x: x.view(-1, 256))
        ).to(device)
        task_model = classifier(256, args.ways).to(device)

    else:
        raise NotImplementedError("Supported datasets are: omniglot, miniimagenet, fc100.")

    print('meta model is :', meta_model.__class__.__name__)
    print('got dataset:', args.dataset)

    # --------------------------------------------------------
    # Resume or start from scratch
    # --------------------------------------------------------
    ckpt_name = f'IFSBA_{args.dataset}_run{run}.pt'
    ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

    if args.resume:
        print('resuming from checkpoint...')
        ckpt = load_checkpoint(ckpt_path=ckpt_path)
        start_iter = ckpt['k']
        total_time = ckpt['total_time']

        meta_model.load_state_dict(ckpt['meta_model'])
        y0 = ckpt['y0']
        w0 = ckpt['w0']

        run_time = ckpt['run_time']
        accs = ckpt['accs']
        vals = ckpt['vals']
        evals = ckpt['evals']
        talist = ckpt['talist']
        run_timea = ckpt['run_timea']
    else:
        print('starting from scratch....')
        start_iter = 0
        total_time = 0.0

        run_time, accs, vals, evals, talist, run_timea = [], [], [], [], [], []

        y0 = [torch.zeros_like(p).to(device) for p in task_model.parameters()]
        w0 = [torch.zeros_like(p).to(device) for p in task_model.parameters()]

    hparams = list(meta_model.parameters())
    meta_bsz = args.batch_size
    inner_log_interval = None

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    for k in range(start_iter, K_outer):
        start_time = time.time()

        y0, w0, train_loss, train_acc, delta_t = train_one_outer_step_ifsba(
            train_tasks=train_tasks,
            meta_model=meta_model,
            task_model=task_model,
            hparams=hparams,
            y0=y0,
            w0=w0,
            reg_param=reg_param,
            shots=args.shots,
            ways=args.ways,
            meta_bsz=meta_bsz,
            lam=args.lmbd,
            K1=args.K1,
            K2=args.K2,
            lr1=lr1,
            lr2=lr2,
            theta1=theta1,
            theta2=theta2,
            sigma=args.sigma,
            M_bound=args.M,
            cheby_K=args.cheby_K,
            l_est=args.l_est,
            mu_est=args.mu_est,
            eps=args.eps,
            device=device,
            inner_log_interval=inner_log_interval,
        )

        step_time = time.time() - start_time
        total_time += step_time

        run_time.append(total_time)
        vals.append(train_loss)
        accs.append(train_acc)

        if (k + 1) % log_interval == 0 or k == 0 or k == K_outer - 1:
            string = (
                f'[Run {run}] IFSBA k={k+1}/{K_outer} '
                f'({step_time:.3f}s) '
                f'Train Loss: {train_loss:.2e}, Train Acc: {100.0*train_acc:.2f}, '
                f'Delta_t: {delta_t:.3e}.'
            )
            out_file.write(string + '\n')
            out_file.flush()
            print(string)

        if (k + 1) % args.save_every == 0:
            state_dict = {
                'k': k + 1,
                'total_time': total_time,
                'meta_model': meta_model.state_dict(),
                'y0': y0,
                'w0': w0,
                'run_time': run_time,
                'accs': accs,
                'vals': vals,
                'evals': evals,
                'talist': talist,
                'run_timea': run_timea,
            }
            save_checkpoint(state_dict, ckpt_path)

        if (k + 1) == stop_k:
            state_dict = {
                'k': k + 1,
                'total_time': total_time,
                'meta_model': meta_model.state_dict(),
                'y0': y0,
                'w0': w0,
                'run_time': run_time,
                'accs': accs,
                'vals': vals,
                'evals': evals,
                'talist': talist,
                'run_timea': run_timea,
            }
            save_checkpoint(state_dict, ckpt_path)
            print('exiting...')
            out_file.close()
            return

        if (k + 1) % eval_interval == 0:
            val_losses, val_accs = evaluate(
                val_tasks,
                meta_model,
                task_model,
                hparams,
                w0,
                reg_param,
                inner_lr,
                inner_mu,
                T_test,
                args.shots,
                args.ways,
            )
            string = (
                "[Run {}] Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) "
                "[mean (+/- std) over {} tasks]."
            ).format(
                run,
                val_losses.mean(),
                val_losses.std(),
                100.0 * val_accs.mean(),
                100.0 * val_accs.std(),
                len(val_losses)
            )
            out_file.write(string + '\n')
            out_file.flush()
            print(string)

            test_losses, test_accs = evaluate(
                test_tasks,
                meta_model,
                task_model,
                hparams,
                w0,
                reg_param,
                inner_lr,
                inner_mu,
                T_test,
                args.shots,
                args.ways,
            )

            evals.append((
                test_losses.mean(),
                test_losses.std(),
                100.0 * test_accs.mean(),
                100.0 * test_accs.std(),
            ))
            talist.append(100.0 * test_accs.mean())
            run_timea.append(total_time)

            string = (
                "[Run {}] Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) "
                "[mean (+/- std) over {} tasks]."
            ).format(
                run,
                test_losses.mean(),
                test_losses.std(),
                100.0 * test_accs.mean(),
                100.0 * test_accs.std(),
                len(test_losses)
            )
            out_file.write(string + '\n')
            out_file.flush()
            print(string)

    out_file.close()


if __name__ == '__main__':
    main()