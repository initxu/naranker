import os
import time
import random
import argparse
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset import NASBenchDataBase, NASBenchDataset, SplitSubet
from ranker import Transformer
from sampler import ArchSampler
from process import train_epoch, validate, sampler_train_iter, sampler_validate
from utils.setup import setup_seed, setup_logger
from utils.config import get_config
from utils.loss_ops import CrossEntropyLossSoft
from utils.optim import LRScheduler


def get_args():
    parser = argparse.ArgumentParser(description='NAR Training')
    parser.add_argument('--config_file',
                        default='./config.yml',
                        type=str,
                        help='training configuration')
    parser.add_argument('--data_path',
                        default='./data/nasbench101/nasbench_only108_with_vertex_flops_and_params_42362.json',
                        type=str,
                        help='Path to load data')
    parser.add_argument('--save_dir',
                        default='./output',
                        type=str,
                        help='Path to save output')

    args = parser.parse_args()

    return args


def build_arg_and_env(run_args):
    args = get_config(run_args.config_file)

    args.config_file = run_args.config_file
    args.data_path = run_args.data_path

    if not os.path.exists(run_args.save_dir):  # 创建output文件存放各种实验文件夹
        os.makedirs(run_args.save_dir)

    args.save_dir = os.path.join(
        run_args.save_dir,
        args.exp_name + '_' + time.strftime('%Y%m%d%H%M%S', time.localtime()))

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)  # 创建当前实验的文件夹

    return args


def main():
    run_args = get_args()
    args = build_arg_and_env(run_args)

    # setup logger
    logger = setup_logger(save_path=os.path.join(args.save_dir, "train.log"))
    logger.info(args)
    # setup tensorboard
    tb_writer = SummaryWriter(os.path.join(args.save_dir,'tensorboard'))

    # setup global seed
    setup_seed(seed=args.seed)
    logger.info('set global random seed = {}'.format(args.seed))

    # setup cuda device
    if args.is_cuda:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch('cpu')

    # build dataset
    if args.space == 'nasbench':
        database = NASBenchDataBase(args.data_path)
        dataset = NASBenchDataset(database, seed=args.seed)
        trainset = SplitSubet(dataset, list(range(args.train_size)))
        valset = SplitSubet(dataset, list(range(args.train_size, args.train_size + args.val_size)))

    # build dataloader
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=getattr(args, 'drop_last', True),
        num_workers=args.data_loader_workers,
        pin_memory=True)

    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.data_loader_workers,
        pin_memory=True)

    # build loss
    criterion = CrossEntropyLossSoft().cuda(device)

    # build model
    logger.info('Building model with {}'.format(args.ranker))
    ranker = Transformer(
        n_tier=args.ranker.n_tier,
        n_arch_patch=args.ranker.n_arch_patch,
        d_patch=args.ranker.d_patch,
        d_patch_vec=args.ranker.d_patch_vec,
        d_model=args.ranker.d_model,
        d_ffn_inner=args.ranker.d_ffn_inner,
        d_tier_prj_inner=args.ranker.d_tier_prj_inner,
        n_layers=args.ranker.n_layers,
        n_head=args.ranker.n_head,
        d_k=args.ranker.d_k,
        d_v=args.ranker.d_v,
        dropout=args.ranker.dropout,
        n_position=args.ranker.n_position,
        scale_prj=args.ranker.scale_prj)
    ranker.cuda(device)

    # build optimizer and lr_scheduler
    logger.info('Building optimizer and lr_scheduler')
    optimizer = optim.AdamW(
        ranker.parameters(),
        betas=(args.optimizer.beta1,args.optimizer.beta2),
        eps=args.optimizer.eps,
        weight_decay=args.optimizer.weight_decay)
    
    # 由于初始lr由LRScheduler设定，因此lr update要优先于optimizer.step()
    lr_scheduler = LRScheduler(
        optimizer,
        lr_mul=args.lr_scheduler.lr_mul,
        d_model=args.ranker.d_model,
        n_warmup_steps=args.lr_scheduler.n_warmup_steps)

    sampler = ArchSampler(
    top_tier=args.sampler.top_tier,
    last_tier= args.sampler.last_tier,
    batch_factor=args.sampler.batch_factor,
    node_type_dict=dict(args.node_type_dict),
    max_edges=args.max_edges,
    reuse_step=args.sampler.reuse_step,
    )

    # train ranker
    for epoch in range(args.start_epochs, args.ranker_epochs):
        flag = 'Ranker Train'
        train_acc, train_loss, batch_statics_dict = train_epoch(ranker, train_dataloader, criterion, optimizer, lr_scheduler, device, args, logger, tb_writer, epoch, flag)
        tb_writer.add_scalar('{}/epoch_accuracy'.format(flag), train_acc, epoch)
        tb_writer.add_scalar('{}/epoch_loss'.format(flag), train_loss, epoch)

        if (epoch+1) % args.validate_freq == 0:
            with torch.no_grad():
                flag = 'Ranker Validate'
                val_acc, val_loss = validate(ranker, val_dataloader, criterion, device, args, logger, epoch, flag)
                tb_writer.add_scalar('{}/epoch_accuracy'.format(flag), val_acc, epoch)
                tb_writer.add_scalar('{}/epoch_loss'.format(flag), val_loss, epoch)
        
    # train ranker with sampler
    assert args.sampler_epochs > args.ranker_epochs, 'sampler_epochs should be larger than ranker_epochs'
    for epoch in range(args.ranker_epochs, args.sampler_epochs):
        flag = 'Sampler Train'

        sample_size = int(args.sampler.sample_size * (1-args.sampler.noisy_factor))
        kl_thred = [args.sampler.flops_kl_thred, args.sampler.params_kl_thred, args.sampler.n_nodes_kl_thred]
        
        # sample from trainset
        sampled_arch, sampled_arch_datast_idx = sampler.sample_arch(batch_statics_dict, sample_size, trainset, kl_thred, args.sampler.max_trails)
        sampled_arch_datast_idx = [v for _, v in enumerate(sampled_arch_datast_idx) if v != None]
        tb_writer.add_scalar('{}/number_sampled_archs'.format(flag), len(sampled_arch_datast_idx), epoch)
        
        # add noisy
        noisy_len = args.sampler.sample_size - len(sampled_arch_datast_idx)
        noisy_samples = random.choices(list(range(args.train_size)), k=noisy_len)
        sampled_arch_datast_idx += noisy_samples
        
        random.shuffle(sampled_arch_datast_idx) # in_place
        assert len(sampled_arch_datast_idx) == args.sampler.sample_size, 'Not enough sampled batch'

        sampled_trainset = SplitSubet(dataset, sampled_arch_datast_idx)
        sampled_train_dataloader = torch.utils.data.DataLoader(
            sampled_trainset,
            batch_size=args.sampler.sample_size,
            num_workers=args.data_loader_workers,
            pin_memory=True)

        val_dataloader = torch.utils.data.DataLoader(
            valset,
            batch_size=args.sampler.sample_size,
            shuffle=False,
            drop_last=True,
            num_workers=args.data_loader_workers,
            pin_memory=True)
        
        train_acc, train_loss, batch_statics_dict = sampler_train_iter(ranker, sampled_train_dataloader, criterion, optimizer, lr_scheduler, device, args, logger, tb_writer, epoch, flag)

        if (epoch+1) % args.validate_freq == 0:
            with torch.no_grad():
                flag = 'Sampler Validate'
                val_acc, val_loss = sampler_validate(ranker, val_dataloader, criterion, device, args, logger, epoch, flag)


        
        



if __name__ == '__main__':
    main()