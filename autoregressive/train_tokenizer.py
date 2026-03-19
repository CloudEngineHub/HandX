import os
import json

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
import models.vqvae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_VQ, dataset_TM_eval
import utils.eval_trans as eval_trans
import warnings
warnings.filterwarnings('ignore')
from collections import defaultdict, OrderedDict
from accelerate.utils import tqdm

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
def def_value():
    return 0.0
assert torch.cuda.is_available()
accelerator = Accelerator(log_with="wandb" )

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
NAME = f'rloss0_{args.exp_name}_patchify_{args.nb_code}_{args.input_dim}_new8'
accelerator.init_trackers(
            project_name="VQ_Training_1d", 
            # config=opt,
            init_kwargs={"wandb": {"name":NAME
                                    ,"resume":'allow',"id":NAME,"save_code":True,}}
            )
args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
args.nb_joints = 22


logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')
comp_device = accelerator.device

##### ---- Dataloader ---- #####
train_loader, train_mean, train_std = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        args.input_dim, 
                                        args.text_type,
                                        args.version, 
                                        'train', 
                                        args.debug,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        num_workers=args.num_workers)
train_mean_tensor = torch.from_numpy(train_mean).float().to(accelerator.device)
train_std_tensor = torch.from_numpy(train_std).float().to(accelerator.device)


val_loader, test_mean, test_std = dataset_TM_eval.MotionMillionFSQDATALoader(args.input_dim, True,
                                        4,
                                        None,
                                        unit_length=2**args.down_t,
                                        version=args.version)

##### ---- Network ---- #####
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                   args.nb_code,
                   args.code_dim,
                   args.output_emb_width,
                   args.down_t,
                   args.stride_t,
                   args.width,
                   args.depth,
                   args.dilation_growth_rate,
                   args.vq_act,
                   args.vq_norm,
                   args.kernel_size,
                   args.use_patcher,
                   args.patch_size,
                   args.patch_method,
                   args.use_attn)

if os.path.exists(os.path.join(args.out_dir, f'net_latest.pth')):
    args.resume_pth = os.path.join(args.out_dir, f'net_latest.pth')
    
if args.resume_pth:
    
    # if os.path.join(args.out_dir, f'net_latest.pth')
     
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    state_dict = torch.load(args.resume_pth, map_location='cpu')
    ckpt = state_dict["net"]
    ckpt = {k.replace('module.', ''): v for k, v in ckpt.items()}
    net.load_state_dict(ckpt, strict=True)
net.train()
net.to(comp_device)

##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5*args.total_iter),int(0.75*args.total_iter)], gamma=args.gamma)

if args.resume_pth:
    optimizer.load_state_dict(state_dict["optimizer"])
    scheduler.load_state_dict(state_dict["scheduler"])

net, optimizer, train_loader, val_loader, scheduler = accelerator.prepare(
        net, optimizer, train_loader, val_loader, scheduler
    )
print(len(train_loader),len(val_loader),'NUM_LOADING')
train_loader_iter = cycle(train_loader)

Loss = losses.Loss_witer(args.recons_loss, args.nb_joints,args.input_dim)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.
logs = defaultdict(def_value, OrderedDict())
log_every = len(train_loader)//3
best_mpjpe = 1000
if args.resume_pth:
    start_iter = state_dict["nb_iter"] + 1
else:
    start_iter = 1

# nb_iter = start_iter
# for gt_motion in tqdm(train_loader):
if accelerator.is_main_process:
    progress_bar = tqdm(range(start_iter, args.total_iter + 1))
for nb_iter in (range(start_iter, args.total_iter + 1)):
    if accelerator.is_main_process:
        progress_bar.update(1)
    if nb_iter%1000 ==0 and accelerator.is_main_process:
        progress_bar.set_postfix(step=nb_iter)
    if not args.resume_pth and nb_iter < args.warm_up_iter:
        # if nb_iter == args.warm_up_iter:
        #     break
        optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
# for nb_iter in tqdm(range(start_iter, args.total_iter + 1)):
    nb_iter = nb_iter+1
    # if nb_iter == args.total_iter + 1:
    #     break
    gt_motion = next(train_loader_iter)
    # gt_motion = gt_motion[...,-20*2*3*2:]
    gt_motion = gt_motion.float().to(comp_device)
    
    pred_motion, loss_commit, perplexity, activate, _ = net(gt_motion)
    loss_motion = Loss(pred_motion, gt_motion)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion)
    
    if args.quantizer == "LFQ":
        loss = loss_motion + loss_commit + args.loss_vel * loss_vel
    else:
        loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel
    
    # if args.use_acc_loss:
    #     loss_acc = Loss.forward_acc(pred_motion, gt_motion)
    #     loss = loss + args.acc_loss * loss_acc
    # if args.use_acc_vel_loss and 1 :
    loss_acc_vel = Loss.forward_acc_vel(pred_motion, gt_motion)
    loss = loss + args.acc_vel_loss * loss_acc_vel
    # if args.use_root_loss and 1 :
    loss_root = Loss.forward_root(pred_motion, gt_motion)
    # loss_root_rot_vel = Loss.forward_root_rot_vel(pred_motion, gt_motion)
    loss = loss + args.root_loss * loss_root
    # loss = loss + args.root_rot_vel_loss * loss_root_rot_vel
    pred_motion_unnorm = pred_motion*train_std_tensor+train_mean_tensor
    gt_motion_unnorm = gt_motion*train_std_tensor+train_mean_tensor
    loss_root_rot,loss_root_rot_vel= Loss.forward_root_rot(pred_motion_unnorm, gt_motion_unnorm)
    loss_root_rot_l1s = Loss.forward_root_rot_l1s(pred_motion, gt_motion)
    
    loss = loss + args.root_rot_vel_loss * args.root_rot_loss*loss_root_rot_vel
    loss = loss + args.root_rot_loss * loss_root_rot
    
    loss = loss + args.root_l1s_loss * loss_root_rot_l1s
    loss_bn = Loss.forward_bl(pred_motion_unnorm, gt_motion_unnorm)
    loss = loss + args.bn_loss * loss_bn
    loss_rel_vel_unnorm = Loss.forward_rel_vel_unnorm(pred_motion_unnorm, gt_motion_unnorm)
    loss = loss + args.rel_vel_loss_unnorm * loss_rel_vel_unnorm
    
    loss_rel_vel = Loss.forward_rel_vel(pred_motion, gt_motion)
    loss = loss + args.rel_vel_loss * loss_rel_vel
    
    loss_vt = Loss.forward_vt(pred_motion_unnorm, gt_motion_unnorm)
    loss_tt = Loss.forward_tt(pred_motion_unnorm, gt_motion_unnorm)
    loss_vel_unnorm = Loss.forward_vel_unnorm(pred_motion_unnorm, gt_motion_unnorm)
    # pred_motion_unnorm = pred_motion*train_std_tensor+train_mean_tensor
    # gt_motion_unnorm = gt_motion*train_std_tensor+train_mean_tensor
    # pred_global_pos=recover_from_local_position_torch(pred_motion_unnorm)
    # gt_global_pos=recover_from_local_position_torch(gt_motion_unnorm)
    
    # Loss.forward_contact_inner(pred_motion_unnorm, gt_motion_unnorm)
    # Loss.forward_contact_between(pred_global_pos, gt_global_pos)
    
    
    
    optimizer.zero_grad()
    accelerator.backward(loss)
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_activate += activate.item()
    logs['loss'] +=loss.item()
    logs['loss_root'] +=loss_root.item()
    logs['loss_acc_vel'] += loss_acc_vel.item()
    logs['loss_vel'] +=loss_vel.item()
    logs['loss_motion'] +=loss_motion.item()
    logs['loss_root_rot'] +=loss_root_rot.item()
    logs['loss_bn'] +=loss_bn.item()
    logs['loss_vt'] +=loss_vt.item()
    logs['loss_tt'] +=loss_tt.item()
    logs['loss_vel_unnorm'] +=loss_vel_unnorm.item()
    logs['loss_root_rot_vel'] +=loss_root_rot_vel.item()
    logs['loss_root_rot_l1s'] +=loss_root_rot_l1s.item()
    logs['loss_rel_vel'] +=loss_rel_vel.item()
    logs['loss_rel_vel_unnorm'] +=loss_rel_vel_unnorm.item()
    
    if nb_iter % log_every == 0 and accelerator.is_main_process:
        mean_loss = OrderedDict()
        # self.logger.add_scalar('val_loss', val_loss, it)
        # self.l
        dict2={}
        for tag, value in logs.items():
            dict2['Train/%s'%tag] = value / log_every
            # self.logger.add_scalar('Train/%s'%tag, value / self.opt.log_every, it)
            # mean_loss[tag] = value / self.opt.log_every
        accelerator.log(dict2, step=nb_iter)

        logs = defaultdict(def_value, OrderedDict())
        avg_perplexity /= log_every
        avg_activate /= log_every
        avg_recons /= log_every
        logger.info(f"Warmup. Iter {nb_iter} :  \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate. {avg_activate:.2f}")

        # if nb_iter % args.print_iter ==  0 :
        #     if accelerator.is_main_process:
        #         avg_recons /= args.print_iter
        #         avg_perplexity /= args.print_iter
        #         avg_commit /= args.print_iter
        #         avg_activate /= args.print_iter
                
                # logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate. {avg_activate:.2f}")
            
        avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.

    # if nb_iter % args.print_iter ==  0 :
    #     if accelerator.is_main_process:
    #         avg_recons /= args.print_iter
    #         avg_perplexity /= args.print_iter
    #         avg_commit /= args.print_iter
    #         avg_activate /= args.print_iter
            
    #         writer.add_scalar('./Train/L1', avg_recons, nb_iter)
    #         writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
    #         writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
    #         writer.add_scalar('./Train/Activate', avg_activate, nb_iter)
            
    #         logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Activate {avg_activate:.2f}")
        
    #     avg_recons, avg_perplexity, avg_commit, avg_activate = 0., 0., 0., 0.
    # args.eval_iter = 3000
    if nb_iter % args.eval_iter==0: #and accelerator.is_main_process:
        # accelerator.wait_for_everyone()
        best_mpjpe, writer, logger = eval_trans.evaluation_vqvae_motionmillion(args.out_dir, train_loader, val_loader, net, logger, writer, nb_iter, best_mpjpe, comp_device=comp_device, codebook_size=accelerator.unwrap_model(net).vqvae.quantizer.codebook_size, accelerator=accelerator)
    # print('F2')
    accelerator.wait_for_everyone()
    # print('FINISH2')
    # accelerator.wait_for_everyone()
    if nb_iter % args.save_iter == 0 and accelerator.is_main_process:
        torch.save({'net' : net.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'nb_iter' : nb_iter}, os.path.join(args.out_dir, f'net_{nb_iter}.pth'))
    if nb_iter % args.save_latest == 0 and accelerator.is_main_process:
        torch.save({'net' : net.state_dict(), 
                    'optimizer' : optimizer.state_dict(),
                    'scheduler' : scheduler.state_dict(),
                    'nb_iter' : nb_iter}, os.path.join(args.out_dir, f'net_latest.pth'))
    
# accelerator.wait_for_everyone()