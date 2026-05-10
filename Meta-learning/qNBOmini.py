"""
Few-shot meta-learning with adaptation over partial parameters 
"""
import math
import argparse
import time
import collections
import os
import pickle
import copy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision.transforms as Tr
from networks import OmniglotNetFeats, MiniimageNetFeats, classifier
from PIL import Image

from utils import Lambda, load_checkpoint, save_checkpoint

import higher
import learn2learn as l2l
from learn2learn.data.transforms import FusedNWaysKShots, LoadData, RemapLabels, ConsecutiveLabels
from sklearn.utils.extmath import safe_sparse_dot

import hypergrad as hg


def split_into_adapt_eval(batch,
               shots,
               ways,
               device=None):

    # Splits task data into adaptation/evaluation sets

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
    Handles the train and validation loss for a single task
    """
    def __init__(self, reg_param, meta_model, task_model, data, batch_size=None): # here batchsize = number of tasks used at each step. we will do full GD for each task
        device = next(meta_model.parameters()).device

        # stateless version of meta_model
        self.fmeta = higher.monkeypatch(meta_model, device=device, copy_initial_weights=True)
        self.ftask = higher.monkeypatch(task_model, device=device, copy_initial_weights=True)

        #self.n_params = len(list(meta_model.parameters()))
        self.train_input, self.train_target, self.test_input, self.test_target = data
        self.reg_param = reg_param
        self.batch_size = 1 if not batch_size else batch_size
        self.val_loss, self.val_acc = None, None

    def compute_feats(self, hparams):
        # compute train feats
        self.train_feats = self.fmeta(self.train_input, params= hparams)

    def reg_f(self, params):
        # l2 regularization
        return sum([(p ** 2).sum() for p in params])

    def train_loss_f(self, params):
        # regularized cross-entropy loss
        out = self.ftask(self.train_feats, params=params)
        return F.cross_entropy(out, self.train_target) + 0.5 * self.reg_param * self.reg_f(params)

    def val_loss_f(self, params, hparams):
        # cross-entropy loss (uses only the task-specific weights in params
        feats = self.fmeta(self.test_input, params=hparams)
        out = self.ftask(feats, params=params)
        val_loss = F.cross_entropy(out, self.test_target)/self.batch_size
        self.val_loss = val_loss.item()  # avoid memory leaks

        pred = out.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        self.val_acc = pred.eq(self.test_target.view_as(pred)).sum().item() / len(self.test_target)

        return val_loss


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

def bfgs(task,hparams, params,tol,step,maxiter_hg,stepsize1,stepsize2,m,params0=None,h0=0.01,device1=None):
            if params0 is not None:
                for param, param0 in zip(params, params0):
                    param.data = param0.data

            task.compute_feats(hparams) # compute feats only once to make inner iterations lighter (only linear transformations!)

            y_list, s_list, mu_list = [], [], []
            y_list1, s_list1, mu_list1 = [], [], []
            y1_list, s1_list, mu1_list = [], [], [] 
            y2_list, s2_list, mu2_list = [], [], [] 
            loss=task.train_loss_f(params)
            ygrad=torch.autograd.grad(loss, params, create_graph=True)
            for k in range(1, step + 1):
                if k<3:
                   params = [w - stepsize1 * (g if g is not None else 0) for w,g in zip(params,ygrad)]
                   loss=task.train_loss_f(params)
                   ygrad=torch.autograd.grad(loss, params, create_graph=True)
                   ngrad1=ygrad[0].detach().cpu().numpy()
                   ngrad1=np.squeeze(ngrad1)
                   ngrad2=ygrad[1].detach().cpu().numpy()
                   ngrad2=np.squeeze(ngrad2)
                else:
                   p1 = two_loops(grady1, m, s_list, y_list, mu_list,h0)
                   p2 = two_loops(grady2, m, s_list1, y_list1, mu_list1,h0)
                   s=stepsize2 * p1
                   s1=stepsize2 * p2
                   st1=torch.from_numpy(s).to(device1).float()
                   st2=torch.from_numpy(s1).to(device1).float()                   
                   params[0]=params[0]+st1
                   params[1]=params[1]+st2
                   loss=task.train_loss_f(params)
                   new_grad=torch.autograd.grad(loss, params, create_graph=True)#
                   ngrad1=new_grad[0].detach().cpu().numpy()
                   ngrad1=np.squeeze(ngrad1)
                   ngrad2=new_grad[1].detach().cpu().numpy()
                   ngrad2=np.squeeze(ngrad2)
                   y=ngrad1-grady1
                   y=np.squeeze(y)
                   s=np.squeeze(s)
                    # Update the memory
                   if (safe_sparse_dot(np.ravel(y),np.ravel(s)))>1e-10:
                      mu=1/safe_sparse_dot(np.ravel(y),np.ravel(s))
                      y_list.append(y.copy())
                      s_list.append(s.copy())
                      mu_list.append(mu)
                   if len(y_list) > m:
                      y_list.pop(0)
                      s_list.pop(0)
                      mu_list.pop(0)
                   y1=ngrad2-grady2
                   y1=np.squeeze(y1)
                   s1=np.squeeze(s1)
                    # Update the memory
                   if (safe_sparse_dot(np.ravel(y1),np.ravel(s1)))>1e-10:
                      y_list1.append(y1.copy())
                      s_list1.append(s1.copy())
                      mu1=1/safe_sparse_dot(np.ravel(y1),np.ravel(s1))
                      mu_list1.append(mu1)
                   if len(y_list) > m:
                      y_list1.pop(0)
                      s_list1.pop(0)
                      mu_list1.pop(0)
                grady1=ngrad1
                grady2=ngrad2
                l_inf_norm_grad = np.linalg.norm(np.ravel(grady1))+np.linalg.norm(np.ravel(grady2))
                if l_inf_norm_grad < tol:
                    break
            o_loss = task.val_loss_f(params,hparams)  # F
            ogrady = torch.autograd.grad(o_loss, params,create_graph=True)# dy F
            gradFy1=ogrady[0].detach().cpu().numpy()#\nabla_y F(x_k,y_{k+1})
            gradFy1=np.squeeze(gradFy1)
            gradFy2=ogrady[1].detach().cpu().numpy()#\nabla_y F(x_k,y_{k+1})
            gradFy2=np.squeeze(gradFy2)
            params1=[w.data.clone() for w in params]
            params1 = [p1.requires_grad_(True) for p1 in params1]
            et=[w.data.clone() for w in params]
            for i in range (1, maxiter_hg + 1):
                hg1= - two_loops(gradFy1, m, s1_list, y1_list, mu1_list,h0)
                hg2= - two_loops(gradFy2, m, s2_list, y2_list, mu2_list,h0)
                #hg = hg/ np.linalg.norm(hg)
                et1=torch.from_numpy(hg1).to(device1).float()
                et2=torch.from_numpy(hg2).to(device1).float()
                params1[0]=params1[0]+et1
                params1[1]=params1[1]+et2
                f1=task.train_loss_f(params1)#f(y_k+e)
                grad=torch.autograd.grad(f1, params1, create_graph=True)#params or params1
                f1grad=grad[0].detach().cpu().numpy()
                f1grad=np.squeeze(f1grad)
                f1grad1=grad[1].detach().cpu().numpy()
                f1grad1=np.squeeze(f1grad1)
                y_tilde1 = f1grad- grady1
                y_tilde2 = f1grad1- grady2
                if safe_sparse_dot(np.ravel(y_tilde1), np.ravel(hg1))>1e-10:
                   y1_list.append(y_tilde1.copy())
                   s1_list.append(hg1.copy())
                   mu1 = 1 / safe_sparse_dot(np.ravel(y_tilde1), np.ravel(hg1))
                   mu1_list.append(mu1)
                if safe_sparse_dot(np.ravel(y_tilde2), np.ravel(hg2))>1e-10:
                   y2_list.append(y_tilde2.copy())
                   s2_list.append(hg2.copy())
                   mu2 = 1 / safe_sparse_dot(np.ravel(y_tilde2), np.ravel(hg2))
                   mu2_list.append(mu2)
                et[0]=et1
                et[1]=et2
                
            #print(f'{k} iterates')
            return params, et

def two_loops(grad_y, m, s_list, y_list, mu_list,h0):
            q = grad_y.copy()
            alpha_list = []
            for s, y, mu in zip(reversed(s_list), reversed(y_list), reversed(mu_list)):
                alpha = mu * safe_sparse_dot(np.ravel(s), np.ravel(q))
                alpha_list.append(alpha)
                q -= alpha * y
            r=h0*q
            for s, y, mu, alpha in zip(s_list, y_list, mu_list, reversed(alpha_list)):
                beta = mu * safe_sparse_dot(np.ravel(y), np.ravel(r))
                r += (alpha - beta) * s
            return -r
        
            
def main():

    parser = argparse.ArgumentParser(description='MAML with Partial Parameter Adaptation')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='miniimagenet', metavar='N', help='omniglot or miniimagenet or fc100')
    parser.add_argument('--resume', type=bool, default=False, help='whether to resume from checkpoint')
    parser.add_argument('--ckpt_dir', type=str, default='metalogs', help='path of checkpoint file')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16, help='meta batch size')
    parser.add_argument('--ways', type=int, default=5, help='num classes in few shot learning')
    parser.add_argument('--shots', type=int, default=5, help='num training shots in few shot learning')
    parser.add_argument('--steps', type=int, default=6000, help='total number of outer steps')
    parser.add_argument('--use_resnet', type=bool, default=False, help='whether to use resnet12 network for minimagenet dataset')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

    args = parser.parse_args()

    if not os.path.isdir(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    run = 1
    mu = 0.1
    inner_lr = .01
    outer_lr = .01
    inner_mu = 0.9
    K = args.steps
    stop_k = None  # stop iteration for early stopping. leave to None if not using it
    n_tasks_train = 20000
    n_tasks_test = 200  # usually around 1000 tasks are used for testing
    n_tasks_val = 200

    if args.dataset == 'omniglot':
        reg_param = 0.2  # reg_param = 2.
        T = 50  # T = 16
    elif args.dataset == 'miniimagenet':
        reg_param = 0.5  # reg_param = 0.5
        T = 30 # T = 30
    elif args.dataset == 'fc100':
        reg_param = 0.5  # reg_param = 0.5
        T = 30 # T = 30

    else:
        raise NotImplementedError(args.dataset, " not implemented!")

    T_test = T
    log_interval = 25
    eval_interval = 50

    loc = locals()
    del loc['parser']
    del loc['args']

    args.out_file = open(os.path.join(args.ckpt_dir, 'FOA'+ args.dataset +'0510'+'.txt'), 'w')

    string = "+++++++++++++++++++ Arguments ++++++++++++++++++++\n"
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
    device = torch.device("cuda:0" if cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

    torch.random.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.dataset == 'omniglot':
        train_tasks, val_tasks, test_tasks = l2l.vision.benchmarks.get_tasksets('omniglot',
                                                      train_ways=args.ways,
                                                      train_samples=2 * args.shots,
                                                      test_ways=args.ways,
                                                      test_samples=2 * args.shots,
                                                      num_tasks=10000,
                                                      root='data/omniglot')
        meta_model = OmniglotNetFeats(64).to(device)
        task_model = classifier(64, args.ways).to(device)

    elif args.dataset == 'miniimagenet':

        MEAN = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        STD = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = Tr.Normalize(mean=MEAN, std=STD)

        # use the same data-augmentation as in lee et al.
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
            download=False)
        #print('got train dataset...')
        val_dataset = l2l.vision.datasets.MiniImagenet(
            root='data/MiniImageNet',
            mode='validation',
            transform=transform_test,
            download=False)
        #print('got val dataset...')
        test_dataset = l2l.vision.datasets.MiniImagenet(
            root='data/MiniImageNet',
            mode='test',
            transform=transform_test,
            download=False)
        #print('got test dataset...')

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
            download=True)

        val_dataset = l2l.vision.datasets.FC100(
            root='data/FC100',
            transform=Tr.ToTensor(),
            mode='validation',
            download=True)

        test_dataset = l2l.vision.datasets.FC100(
            root='data/FC100',
            transform=Tr.ToTensor(),
            mode='test',
            download=True)

        meta_model = torch.nn.Sequential(l2l.vision.models.ConvBase(output_size=64, channels=3, max_pool=True),
                                         Lambda(lambda x: x.view(-1, 256))).to(device)
        task_model = classifier(256, args.ways).to(device)
    else:
        raise NotImplementedError("Supported datasets are: omniglot, miniimagenet and fc100.")

    print('meta model is : ', meta_model.__class__.__name__)

    if args.dataset == 'miniimagenet' or args.dataset =='fc100':
        train_dataset = l2l.data.MetaDataset(train_dataset)
        val_dataset = l2l.data.MetaDataset(val_dataset)
        test_dataset = l2l.data.MetaDataset(test_dataset)

        train_transforms = [FusedNWaysKShots(train_dataset, n=args.ways, k=2 * args.shots),
        LoadData(train_dataset),
        RemapLabels(train_dataset),
        ConsecutiveLabels(train_dataset)]

        train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=train_transforms, num_tasks=n_tasks_train)

        val_transforms = [FusedNWaysKShots(val_dataset, n=args.ways, k=2 * args.shots),
        LoadData(val_dataset),
        ConsecutiveLabels(val_dataset),
        RemapLabels(val_dataset)]

        val_tasks = l2l.data.TaskDataset(val_dataset, task_transforms=val_transforms, num_tasks=n_tasks_val)

        test_transforms = [FusedNWaysKShots(test_dataset, n=args.ways, k=2 * args.shots),
        LoadData(test_dataset),
        RemapLabels(test_dataset),
        ConsecutiveLabels(test_dataset)]

        test_tasks = l2l.data.TaskDataset(test_dataset, task_transforms=test_transforms, num_tasks=n_tasks_test)

    print('got dataset: ', args.dataset)

    
    print('starting from scratch....')
    start_iter = 0
    total_time = 0

    run_time, accs, vals, evals,talist,run_timea = [], [], [], [],[],[]

    w0 = [torch.zeros_like(p).to(device) for p in task_model.parameters()]

    hparams = list(meta_model.parameters())

    outer_opt = torch.optim.Adam(params=hparams, lr=outer_lr)


    inner_log_interval = None
    inner_log_interval_test = None

    meta_bsz = args.batch_size

    # training starts here
    for k in range(start_iter, K):
        start_time = time.time()

        outer_opt.zero_grad()

        val_loss, val_acc = 0, 0
        forward_time, backward_time= 0, 0
        w_accum = [torch.zeros_like(w).to(device) for w in w0]

        th = 0.0

        for t_idx in range(meta_bsz):
            start_time_task = time.time()

            # sample a training task
            task_data = train_tasks.sample()

            task_data = split_into_adapt_eval(task_data,
                                              shots=args.shots,
                                              ways=args.ways,
                                              device=device)
            # single task set up
            task = Task(reg_param, meta_model, task_model, task_data, batch_size=meta_bsz)

            # single task inner loop
            params = [p.detach().clone().requires_grad_(True) for p in w0]
            
            final_params,et = bfgs(task, hparams, params, tol=1/((k+1)*(t_idx+1)),step=20, maxiter_hg=3,stepsize1=0.1,stepsize2=0.1,m=30,params0=w0,h0=0.01,device1=device)#T=50

            

            forward_time_task = time.time() - start_time_task


            # single task hypergradient computation
            th0 = time.time()
            
            oloss=task.val_loss_f(final_params,hparams)
            Fx=torch.autograd.grad(oloss,hparams,retain_graph=True,allow_unused=True)

            loss=task.train_loss_f(final_params)
            grad0=torch.autograd.grad(loss,final_params, create_graph=True,retain_graph=True)
            wmap=[(g if g is not None else 0*grad0[0]) for g in grad0]
            Fy=torch.autograd.grad(wmap,hparams,grad_outputs=et,
                                   allow_unused=True)
            
            grads=[F_x-F_y for F_x,F_y in zip(Fx,Fy)]
            update_tensor_grads(hparams, grads)# will accumulate single task hypergradient to get overall hypergradient
            th += time.time() - th0

            backward_time_task = time.time() - start_time_task - forward_time_task

            val_loss += task.val_loss
            val_acc += task.val_acc/task.batch_size

            forward_time += forward_time_task
            backward_time += backward_time_task

            w_accum = [p + fp / meta_bsz for p, fp in zip(w_accum, final_params)]

        outer_opt.step()

        w0 = [w.clone() for w in w_accum]  # will be used as initialization for next step

        step_time = time.time() - start_time
        total_time += step_time

        run_time.append(total_time)
        vals.append(val_loss) # this is actually train loss in few-shot learning
        accs.append(val_acc) # this is actually train accuracy in few-shot learning

        if val_loss > 2.0 and k > 20: # quit if loss goes up after some iterations
            print('loss went up! exiting...')
            #exit()

        if k >= 1500: # 2000
            inner_lr = 0.01
            outer_lr = 0.001
            for param_group in outer_opt.param_groups:
                param_group['lr'] = outer_lr

        if k >= 3500: # 5000
            inner_lr = 0.01
            outer_lr = 0.0001 #0.0005
            for param_group in outer_opt.param_groups:
                param_group['lr'] = outer_lr


        if (k+1) % log_interval == 0 or k == 0 or k == K-1:
            string = 'META k={}/{} Lr: {:.5f} mu: {:.3f}  ({:.3f}s F: {:.3f}s, B: {:.3f}s, HG: {:.3f}s) Train Loss: {:.2e}, Train Acc: {:.2f}.'.format(k+1, K, outer_lr, mu, step_time, forward_time, backward_time, th, val_loss, 100. * val_acc)
            args.out_file.write(string + '\n')
            args.out_file.flush()
            print(string)

        if (k+1) % args.save_every == 0:
            state_dict = {'k': k+1,
                          'acc': accs,
                          'val': vals,
                          'testaccuracy':talist,
                          'eval': evals,
                          'time': run_time,
                          'timea':run_timea,
                          'hp': hparams,
                          'w': w0,
                          'opt': outer_opt.state_dict()
                          }
            filename = 'FOA' + args.dataset +  '0510'+  '.pt'
            save_path = os.path.join(args.ckpt_dir, filename)

            save_checkpoint(state_dict, save_path)

        if (k+1) == stop_k: # early stopping

            state_dict = {'k': k+1,
                          'acc': accs,
                          'val': vals,
                          'testaccuracy':talist,
                          'eval': evals,
                          'time': run_time,
                          'timea':run_timea,
                          'hp': hparams,
                          'w': w0,
                          'opt': outer_opt.state_dict()
                          }
            filename = 'FOA' + args.dataset  +  '0510'+  '.pt'
            save_path = os.path.join(args.ckpt_dir, filename)

            save_checkpoint(state_dict, save_path)
            print('exiting...')
            exit()

        if (k+1) % eval_interval == 0:
            val_losses, val_accs = evaluate(val_tasks, meta_model, task_model, hparams, w0, reg_param,
                                              inner_lr, inner_mu, T_test, args.shots, args.ways)

            #evals.append((val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std()))
            string = "Val loss {:.2e} (+/- {:.2e}): Val acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(val_losses.mean(), val_losses.std(), 100. * val_accs.mean(), 100. * val_accs.std(), len(val_losses))
            args.out_file.write(string + '\n')
            args.out_file.flush()
            print(string)

            test_losses, test_accs = evaluate(test_tasks, meta_model, task_model, hparams, w0, reg_param,
                                              inner_lr, inner_mu, T_test, args.shots, args.ways)

            evals.append((test_losses.mean(), test_losses.std(), 100. * test_accs.mean(), 100.*test_accs.std()))
            ta=100. * test_accs.mean()
            talist.append(ta)
            run_timea.append(total_time)

            string = "Test loss {:.2e} (+/- {:.2e}): Test acc: {:.2f} (+/- {:.2e}) [mean (+/- std) over {} tasks].".format(test_losses.mean(), test_losses.std(), 100. * test_accs.mean(),100.*test_accs.std(), len(test_losses))
            args.out_file.write(string + '\n')
            args.out_file.flush()
            print(string)


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
        final_params, _ = bfgs(
                                task,
                                hparams,
                                params,
                                tol=1e-3,
                                step=inner_steps,
                                maxiter_hg=1,
                                stepsize1=inner_lr,
                                stepsize2=inner_lr,
                                m=5,
                                params0=w0,
                                h0=0.01,
                                device1=device
                            )

        task.val_loss_f(final_params, hparams)

        eval_losses.append(task.val_loss)
        eval_accs.append(task.val_acc)

        if k >= 999: # use at most 1000 tasks for evaluation
            return np.array(eval_losses), np.array(eval_accs)

    return np.array(eval_losses), np.array(eval_accs)


def update_tensor_grads(params, grads):
    for l, g in zip(params, grads):
        if l.grad is None:
            l.grad = torch.zeros_like(l)
        if g is not None:
            l.grad += g


if __name__ == '__main__':
    main()
