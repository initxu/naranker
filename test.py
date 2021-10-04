import os
import torch
import argparse
import torch.utils.data

from dataset import NASBenchDataBase, NASBenchDataset, SplitSubet
from architecture import Bucket
from ranker import Transformer
from sampler import ArchSampler
from utils.metric import AverageMeter
from utils.loss_ops import CrossEntropyLossSoft
from utils.config import get_config
from utils.setup import setup_seed, setup_logger
from process import validate, evaluate_sampled_batch
from process.train_utils import init_tier_list


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
                        default='./output/bestarch_distri_wnoisy_20211004173337',
                        type=str,
                        help='Path to save output')
    parser.add_argument('--checkpoint',
                        default='ckp_best.pth.tar',
                        type=str,
                        help='checkpoint file')

    args = parser.parse_args()

    return args

def main():
    run_args = get_args()
    
    args = get_config(run_args.config_file)
    args.config_file = run_args.config_file
    args.data_path = run_args.data_path
    args.save_dir = run_args.save_dir
    
    ckp_path = os.path.join(run_args.save_dir, run_args.checkpoint)
    assert os.path.isfile(ckp_path), 'Checkpoint file does not exist at {}'.format(ckp_path)
    with open(ckp_path, 'rb') as f:
        checkpoint = torch.load(f, map_location=torch.device('cpu'))

    logger = setup_logger(save_path=os.path.join(args.save_dir, "test.log"))
    logger.info(args)

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
        valset = SplitSubet(dataset, list(range(args.train_size, args.train_size + args.val_size)))
        
    # build dataloader
    val_dataloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=args.data_loader_workers,
        pin_memory=True)

    criterion = CrossEntropyLossSoft().cuda(device)

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
    
    ranker.load_state_dict(checkpoint['state_dict'])
    ranker.cuda(device)

    sampler = ArchSampler(
    top_tier=args.sampler.top_tier,
    last_tier= args.sampler.last_tier,
    batch_factor=args.sampler.batch_factor,
    node_type_dict=dict(args.node_type_dict),
    max_edges=args.max_edges,
    reuse_step=args.sampler.reuse_step,
    )

    with torch.no_grad():
        flag = 'Ranker Test'
        val_acc, val_loss = validate(ranker, val_dataloader, criterion, device, args, logger, 0, flag)

    # sample
    assert args.sampler_epochs > args.ranker_epochs, 'sampler_epochs should be larger than ranker_epochs'
    assert Bucket.get_n_tier()==0, 'Bucket counts should be reset to 0'
    tier_list = init_tier_list(args)
    
    history_best_acc = 0
    history_best_arch_iter = 0
    history_best_rank=0
    history_best_distri = checkpoint['batch_distri']
    sampled_arch_acc = AverageMeter()
    for it in range(args.ranker_epochs, args.sampler_epochs):
        flag = 'Sample Test'
        
        with torch.no_grad():
            batch_statics_dict, (acc, rank) = evaluate_sampled_batch(ranker, sampler, tier_list, history_best_distri, dataset, it, args, device, None, logger, flag)
            sampled_arch_acc.update(acc, n=1)
            
            if acc > history_best_acc:
                history_best_arch_iter = it - args.ranker_epochs
                history_best_acc = acc
                history_best_rank = rank
                logger.info('[Best] Found History Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:4d}(top {:.2%})'.format(
                    history_best_arch_iter,
                    history_best_acc,  
                    history_best_rank,
                    history_best_rank/len(dataset)))
            # else:
                # history_best_distri = batch_statics_dict
                
    logger.info('[Result] Derive History Best Arch in Iter {:2d}: Test Acc {:.8f} Rank: {:4d}(top {:.2%}), Avg Test Acc {:.8f}'.format(
        history_best_arch_iter,
        history_best_acc,  
        history_best_rank,
        history_best_rank/len(dataset),
        sampled_arch_acc.avg))
    

if __name__ == '__main__':
        main()


