#!/usr/bin/env python3
"""
Few-shot meta-learning with adaptation over partial parameters
F2BA version
"""

import math
import argparse
import time
import collections
import os
import pickle

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as Tr

from networks import OmniglotNetFeats, MiniimageNetFeats, classifier
from utils import Lambda, load_checkpoint, save_checkpoint

import higher
import learn2learn as l2l
from learn2learn.data.transforms import (
    FusedNWaysKShots,
    LoadData,
    RemapLabels,
    ConsecutiveLabels,
)

# from networks import ResNet12


def split_into_adapt_eval(batch, shots, ways, device=None):
    """
    Splits task data into adaptation/evaluation sets.
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


class Task:
    """
    Handles the support/query loss for a single few-shot task.
    x := hparams (meta/backbone params)
    y,z := task-specific classifier params
    g := support/adaptation loss
    f := query/evaluation loss
    """

    def __init__(self, reg_param, meta_model, task_model, data, batch_size=None):
        device = next(meta_model.parameters()).device

        # functional versions
        self.fmeta = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)
        self.ftask = higher.monkeypatch(task_model, device=device, copy_initial_weights=True)

        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def compute_feats(self, hparams):
        # compute train feats
        self.train_feats = self.fmeta(self.train_input, params= hparams)

    def reg_f(self, params):
        """
        L2 regularization on task-specific params.
        """
        return sum((p ** 2).sum() for p in params)

    def g_loss(self, hparams, params, divide_by_batch=False):
        """
        g(x,y): support/adaptation loss
        """
        feats = self.fmeta(self.train_input, params=hparams)
        out = self.ftask(feats, params=params)
        loss = F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.reg_f(params)
        if divide_by_batch:
            loss = loss / self.batch_size
        return loss

    def f_loss(self, hparams, params, divide_by_batch=False, record=False):
        """
        f(x,y): query/evaluation loss
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

    def train_loss_f(self, params):
        # regularized cross-entropy loss
        out = self.ftask(self.train_feats, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.reg_f(params)

    def val_loss_f(self, params, hparams):
        return self.f_loss(hparams, params, divide_by_batch=True, record=True)


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
    """
    Accumulate grads into params[i].grad so that optimizer.step() can use them.
    """
    for p, g in zip(params, grads):
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if g is not None:
            p.grad += g


def F2BA_inner(task, hparams, w0, alpha, tau, lam, K_inner, log_interval=None):
    """
    F2BA inner loop:
        z^0 = y^0 = w0
        z^{k+1} = z^k - alpha * ∇_y g(x, z^k)
        y^{k+1} = y^k - tau * ( ∇_y f(x, y^k) + lam * ∇_y g(x, y^k) )
    """
    y = clone_params(w0, requires_grad=True)
    z = clone_params(w0, requires_grad=True)

    for k in range(K_inner):
        # z-update
        g_z = task.g_loss(hparams, z, divide_by_batch=False)
        grad_z = safe_grads(g_z, z, create_graph=False, retain_graph=False)
        with torch.no_grad():
            z_next = [p - alpha * g for p, g in zip(z, grad_z)]
        z = [p.detach().requires_grad_(True) for p in z_next]

        # y-update
        f_y = task.f_loss(hparams, y, divide_by_batch=False, record=False)
        g_y = task.g_loss(hparams, y, divide_by_batch=False)
        grad_f_y = safe_grads(f_y, y, create_graph=False, retain_graph=False)
        grad_g_y = safe_grads(g_y, y, create_graph=False, retain_graph=False)

        with torch.no_grad():
            y_next = [
                p - tau * (gf + lam * gg)
                for p, gf, gg in zip(y, grad_f_y, grad_g_y)
            ]
        y = [p.detach().requires_grad_(True) for p in y_next]

        if log_interval and (k % log_interval == 0 or k == K_inner - 1):
            with torch.no_grad():
                cur_f = task.f_loss(hparams, y, divide_by_batch=False, record=False).item()
                cur_g = task.g_loss(hparams, y, divide_by_batch=False).item()
            print(f'F2BA inner k={k}, f={cur_f:.6f}, g={cur_g:.6f}')

    # detach here: outer gradient uses partial-x derivatives at final yK,zK
    yK = detach_params(y, requires_grad=False)
    zK = detach_params(z, requires_grad=False)
    return yK, zK

def F2BA_hypergrad(task, hparams, yK, zK, lam):
    """
    Compute
        grad_hat = ∇_x f(x, yK) + lam * ( ∇_x g(x, yK) - ∇_x g(x, zK) )
    """
    f_val = task.f_loss(hparams, yK, divide_by_batch=True, record=True)
    g_y_val = task.g_loss(hparams, yK, divide_by_batch=True)
    g_z_val = task.g_loss(hparams, zK, divide_by_batch=True)

    grad_f_x = safe_grads(f_val, hparams, create_graph=False, retain_graph=False)
    grad_g_y_x = safe_grads(g_y_val, hparams, create_graph=False, retain_graph=False)
    grad_g_z_x = safe_grads(g_z_val, hparams, create_graph=False, retain_graph=False)

    grad_hat = [
        gf + lam * (ggy - ggz)
        for gf, ggy, ggz in zip(grad_f_x, grad_g_y_x, grad_g_z_x)
    ]
    return grad_hat


def inner_solver(task, hparams, params, steps, optim, params0=None, log_interval=None):

    if params0 is not None:
        for param, param0 in zip(params, params0):
            param.data = param0.data

    task.compute_feats(hparams) # compute feats only once to make inner iterations lighter (only linear transformations!)

    for t in range(steps):
        loss = task.train_loss_f(params)
        optim.zero_grad()
        grads = torch.autograd.grad(loss, params)
        update_tensor_grads(params, grads)
        optim.step()

        if log_interval and (t % log_interval==0 or t==steps-1):
            print('Inner step t={}, Loss: {:.6f}'.format(t, loss.item()))

    return [param.detach().clone() for param in params]

def evaluate(metadataset, meta_model, task_model, hparams, w0, reg_param, inner_lr, inner_mu, inner_steps, shots, ways):
    #meta_model.train()
    device = next(meta_model.parameters()).device

    iters = metadataset.num_tasks
    eval_losses, eval_accs = [], []

    for k in range(iters):

        data = metadataset.sample()
        data = split_into_adapt_eval(data,
                                     shots=shots,
                                     ways=ways,
                                     device=device)

        task = Task(reg_param, meta_model, task_model, data) # metabatchsize will be 1 here

        # single task inner loop
        params = [p.detach().clone().requires_grad_(True) for p in w0]
        inner_opt = torch.optim.SGD(lr=inner_lr, momentum=inner_mu, params=params)
        final_params = inner_solver(task, hparams, params, inner_steps, optim=inner_opt, params0=w0)

        inner_opt.state = collections.defaultdict(dict)  # reset inner optimizer state

        task.val_loss_f(final_params, hparams)

        eval_losses.append(task.val_loss)
        eval_accs.append(task.val_acc)

        if k >= 999: # use at most 1000 tasks for evaluation
            return np.array(eval_losses), np.array(eval_accs)

    return np.array(eval_losses), np.array(eval_accs)


def main():
    parser = argparse.ArgumentParser(description='F2BA with Partial Parameter Adaptation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='fc100',
                        metavar='N', help='omniglot or miniimagenet or fc100')
    parser.add_argument('--run_id', type=int, default=1, help='which run this is, e.g. 1,2,3,4,5')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume from checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='metalogs', help='path of checkpoint file')
    parser.add_argument('--save_every', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5, help='num classes in few-shot learning')
    parser.add_argument('--shots', type=int, default=5, help='num training shots in few-shot learning')
    parser.add_argument('--steps', type=int, default=1000, help='total number of outer steps')
    parser.add_argument('--use_resnet', type=bool, default=False, help='whether to use resnet12 on miniimagenet')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    # F2BA hyperparameters
    parser.add_argument('--f2ba_alpha', type=float, default=0.01, help='z-update step size')
    parser.add_argument('--f2ba_tau', type=float, default=0.1, help='y-update step size')
    parser.add_argument('--f2ba_lambda', type=float, default=1, help='lambda in F2BA')
    parser.add_argument('--f2ba_Kinner', type=int, default=30, help='number of F2BA inner iterations')

    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    run = args.run_id
    outer_lr = 0.01
    inner_lr = 0.01
    inner_mu = 0.9
    K = args.steps
    stop_k = None

    n_tasks_train = 20000
    n_tasks_test = 200
    n_tasks_val = 200

    if args.dataset == 'omniglot':
        reg_param = 0.2
        T = args.f2ba_Kinner
    elif args.dataset == 'miniimagenet':
        reg_param = 0.5
        T = args.f2ba_Kinner
    elif args.dataset == 'fc100':
        reg_param = 0.5
        T = args.f2ba_Kinner
    else:
        raise NotImplementedError(args.dataset, " not implemented!")

    T_test = T
    log_interval = 25
    eval_interval = 50

    alpha = args.f2ba_alpha
    tau = args.f2ba_tau
    lam = args.f2ba_lambda

    loc = locals()
    del loc['parser']
    del loc['args']

    log_name = f'log_F2BA_{args.dataset}_run{run}.txt'
    args.out_file = open(os.path.join(args.ckpt_dir, log_name), 'w')

    string = f"+++++++++++++++++++ Run {run} Arguments ++++++++++++++++++++\n"
    for item, value in args.__dict__.items():
        string += "{}:{}\n".format(item, value)

    args.out_file.write(string + '\n')
    args.out_file.flush()
    print(string + '\n')

    string = ""
    for item, value in loc.items():
        string += "{}:{}\n".format(item, value)

    args.out_file.write(string + '\n')
    args.out_file.flush()
    print(string, '\n')

    cuda = not args.no_cuda and torch.cuda.is_available()
    if cuda:
        print('Training on cuda device...')
    else:
        print('Training on cpu...')
    device = torch.device("cuda" if cuda else "cpu")

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---------------------------------------------------------------------
    # Dataset + model
    # ---------------------------------------------------------------------
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

        transform_train = Tr.Compose([
            # Tr.ToPILImage(),
            # Tr.RandomCrop(84, padding=8),
            # Tr.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # Tr.RandomHorizontalFlip(),
            # Tr.ToTensor(),
            normalize
        ])

        transform_test = Tr.Compose([
            normalize
        ])

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
            meta_model = ResNet12(avg_pool=True, drop_rate=0.0, keep_prob=1.0).to(device)
            task_model = classifier(640, args.ways).to(device)
        else:
            meta_model = MiniimageNetFeats(32).to(device)
            task_model = classifier(32 * 5 * 5, args.ways).to(device)

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

        meta_model = torch.nn.Sequential(
            l2l.vision.models.ConvBase(hidden=64, channels=3, max_pool=True),
            Lambda(lambda x: x.view(-1, 256))
        ).to(device)
        task_model = classifier(256, args.ways).to(device)

    else:
        raise NotImplementedError("Supported datasets are: omniglot, miniimagenet and fc100.")

    print('meta model is : ', meta_model.__class__.__name__)

    if args.dataset in ['miniimagenet', 'fc100']:
        train_dataset = l2l.data.MetaDataset(train_dataset)
        val_dataset = l2l.data.MetaDataset(val_dataset)
        test_dataset = l2l.data.MetaDataset(test_dataset)

        train_transforms = [
            FusedNWaysKShots(train_dataset, n=args.ways, k=2 * args.shots),
            LoadData(train_dataset),
            RemapLabels(train_dataset),
            ConsecutiveLabels(train_dataset)
        ]
        train_tasks = l2l.data.TaskDataset(
            train_dataset,
            task_transforms=train_transforms,
            num_tasks=n_tasks_train
        )

        val_transforms = [
            FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
            LoadData(val_dataset),
            ConsecutiveLabels(val_dataset),
            RemapLabels(val_dataset)
        ]
        val_tasks = l2l.data.TaskDataset(
            val_dataset,
            task_transforms=val_transforms,
            num_tasks=n_tasks_val
        )

        test_transforms = [
            FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
            LoadData(test_dataset),
            RemapLabels(test_dataset),
            ConsecutiveLabels(test_dataset)
        ]
        test_tasks = l2l.data.TaskDataset(
            test_dataset,
            task_transforms=test_transforms,
            num_tasks=n_tasks_test
        )

    print('got dataset: ', args.dataset)

    # ---------------------------------------------------------------------
    # Resume or start from scratch
    # ---------------------------------------------------------------------
    if args.resume:
        print('resuming from checkpoint...')
        filename = 'F2BA_shots5_' + args.dataset + '_T' + str(T) + '_run' + str(run) + '.pt'
        try:
            ckpt = load_checkpoint(ckpt_path=os.path.join(args.ckpt_dir, filename))
            start_iter = ckpt['k']

            accs = ckpt['acc']
            vals = ckpt['val']
            run_time = ckpt['time']
            evals = ckpt['eval']
            total_time = run_time[-1]

            w0 = ckpt['w']

            hparams = ckpt['hp']
            hparams = [hp.detach().requires_grad_(True) for hp in hparams]

            outer_opt = torch.optim.Adam(params=hparams, lr=outer_lr)
            outer_opt.load_state_dict(ckpt['opt'])
        except Exception:
            raise FileNotFoundError('Cannot find checkpoint file')
    else:
        print('starting from scratch....')
        start_iter = 0
        total_time = 0

        run_time, accs, vals, evals, talist, run_timea = [], [], [], [], [], []

        # task-specific initialization
        w0 = [torch.zeros_like(p).to(device) for p in task_model.parameters()]

        # meta parameters
        hparams = list(meta_model.parameters())

        outer_opt = torch.optim.Adam(params=hparams, lr=outer_lr)

    inner_log_interval = None
    meta_bsz = args.batch_size

    # ---------------------------------------------------------------------
    # Training starts here
    # ---------------------------------------------------------------------
    for k in range(start_iter, K):
        start_time = time.time()

        outer_opt.zero_grad()

        val_loss, val_acc = 0.0, 0.0
        forward_time, backward_time = 0.0, 0.0

        # optional: refresh w0 by averaging yK across tasks
        w_accum = [torch.zeros_like(w).to(device) for w in w0]

        th = 0.0

        for t_idx in range(meta_bsz):
            start_time_task = time.time()

            # sample a training task
            task_data = train_tasks.sample()
            task_data = split_into_adapt_eval(
                task_data,
                shots=args.shots,
                ways=args.ways,
                device=device
            )

            task = Task(reg_param, meta_model, task_model, task_data, batch_size=meta_bsz)

            # single-task F2BA inner loop
            yK, zK = F2BA_inner(
                task=task,
                hparams=hparams,
                w0=w0,
                alpha=alpha,
                tau=tau,
                lam=lam,
                K_inner=T,
                log_interval=inner_log_interval,
            )

            forward_time_task = time.time() - start_time_task

            # single-task outer gradient computation
            th0 = time.time()

            grads = F2BA_hypergrad(
                task=task,
                hparams=hparams,
                yK=yK,
                zK=zK,
                lam=lam
            )

            update_tensor_grads(hparams, grads)
            th += time.time() - th0

            backward_time_task = time.time() - start_time_task - forward_time_task

            val_loss += task.val_loss
            val_acc += task.val_acc / task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task

            # use average yK as next initialization
            w_accum = [p + y / meta_bsz for p, y in zip(w_accum, yK)]

        # outer update
        outer_opt.step()

        # refresh classifier initialization
        w0 = [w.clone() for w in w_accum]

        step_time = time.time() - start_time
        total_time += step_time

        run_time.append(total_time)
        vals.append(val_loss)
        accs.append(val_acc)

        if val_loss > 2.0 and k > 20:
            print('loss went up! exiting...')
            exit()

        if k >= 1500:
            outer_lr = 0.001
            for param_group in outer_opt.param_groups:
                param_group['lr'] = outer_lr

        if k >= 3500:
            outer_lr = 0.0001
            for param_group in outer_opt.param_groups:
                param_group['lr'] = outer_lr

        if (k + 1) % log_interval == 0 or k == 0 or k == K - 1:
            string = (
                'META k={}/{} Lr: {:.5f} alpha: {:.5f} tau: {:.5f} lambda: {:.5f}  '
                '({:.3f}s F: {:.3f}s, B: {:.3f}s, HG: {:.3f}s) '
                'Train Loss: {:.2e}, Train Acc: {:.2f}.'
            ).format(
                k + 1, K, outer_lr, alpha, tau, lam,
                step_time, forward_time, backward_time, th,
                val_loss, 100. * val_acc
            )
            args.out_file.write(string + '\n')
            args.out_file.flush()
            print(string)

        if (k + 1) % args.save_every == 0:
            state_dict = {
                'k': k + 1,
                'acc': accs,
                'val': vals,
                'testaccuracy': talist,
                'eval': evals,
                'time': run_time,
                'timea': run_timea,
                'hp': hparams,
                'w': w0,
                'opt': outer_opt.state_dict()
            }
            filename = 'F2BA_shots5_' + args.dataset + '_T' + str(T) + '_run' + str(run) + '.pt'
            save_path = os.path.join(args.ckpt_dir, filename)
            save_checkpoint(state_dict, save_path)

        if (k + 1) == stop_k:
            state_dict = {
                'k': k + 1,
                'acc': accs,
                'val': vals,
                'testaccuracy': talist,
                'eval': evals,
                'time': run_time,
                'timea': run_timea,
                'hp': hparams,
                'w': w0,
                'opt': outer_opt.state_dict()
            }
            filename = 'F2BA_shots5_' + args.dataset + '_T' + str(T) + '_run' + str(run) + '.pt'
            save_path = os.path.join(args.ckpt_dir, filename)
            save_checkpoint(state_dict, save_path)
            print('exiting...')
            exit()

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
                args.ways
                    )

            string = (
                "[Run {}] Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) "
                "[mean (+/- std) over {} tasks]."
            ).format(
                run,
                val_losses.mean(),
                val_losses.std(),
                100. * val_accs.mean(),
                100. * val_accs.std(),
                len(val_losses)
            )
            args.out_file.write(string + '\n')
            args.out_file.flush()
            print(string)

            test_losses, test_accs = evaluate(test_tasks, meta_model, task_model, hparams, w0, reg_param,
                                              inner_lr, inner_mu, T_test, args.shots, args.ways)

            evals.append((
                test_losses.mean(),
                test_losses.std(),
                100. * test_accs.mean(),
                100. * test_accs.std()
            ))
            ta = 100. * test_accs.mean()
            talist.append(ta)
            run_timea.append(total_time)

            string = (
                "[Run {}] Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) "
                "[mean (+/- std) over {} tasks]."
            ).format(
                run,
                test_losses.mean(),
                test_losses.std(),
                100. * test_accs.mean(),
                100. * test_accs.std(),
                len(test_losses)
            )
            args.out_file.write(string + '\n')
            args.out_file.flush()
            print(string)


if __name__ == '__main__':
    main()