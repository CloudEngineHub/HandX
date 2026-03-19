
import os
import torch
from os.path import join as pjoin
import numpy as np
# from models.modules import MovementConvEncoder, TextEncoderBiGRUCo, MotionEncoderBiGRUCo
from utils.word_vectorizer import POS_enumerator


#################Encoding a batch#############

## pred_motions: [B,T,2,21,3] T=60

## Define mean_train,std_train used for training
# :
# mean_train = 
# std_train = 

## Canonicalize: Right root to [0,0,0]

# base
#  ['checkpoint_dir', 'fixed_frames', 'generated_normalized', 'generated_real', 'gt_motion_normalized', 'gt_motion_real', 'guidance_scale', 'model_name', 'motion_length', 'nfeats', 'njoints', 'text_prompt', 'val_index']

from tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
from tma.models.architectures.temos.motionencoder.actor import ActorAgnosticEncoder

_DEFAULT_EVALUATOR_CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'evaluator', 'epoch=269.ckpt')

def build_models(opt, evaluator_ckpt=None):
    path = evaluator_ckpt or os.environ.get('EVALUATOR_CKPT', _DEFAULT_EVALUATOR_CKPT)
    A=torch.load(path,weights_only=False)
    STAT_DICT=A['state_dict']

    filtered_dict = {k[12:]: v for k, v in STAT_DICT.items() if k.startswith("textencoder")}
    filtered_dict2 = {k[14:]: v for k, v in STAT_DICT.items() if k.startswith("motionencoder")}
    #A['state_dict']=filtered_dict
    modelpath = 'distilbert-base-uncased'

    textencoder = DistilbertActorAgnosticEncoder(modelpath,latent_dim=256, 
    ff_size=1024,num_layers=4)
    textencoder.load_state_dict(filtered_dict)

    ## nfeats 231: motion dimension, 128*3: bps dimension
    motionencoder=ActorAgnosticEncoder(nfeats=21*3*2,vae=True,latent_dim=256,ff_size=1024,
                                    num_layers=4)
    motionencoder.load_state_dict(filtered_dict2)
    motionencoder.eval()
    textencoder.eval()
    # movement_enc = MovementConvEncoder(opt.dim_pose-4, opt.dim_movement_enc_hidden, opt.dim_movement_latent)
    # text_enc = TextEncoderBiGRUCo(word_size=opt.dim_word,
    #                               pos_size=opt.dim_pos_ohot,
    #                               hidden_size=opt.dim_text_hidden,
    #                               output_size=opt.dim_coemb_hidden,
    #                               device=opt.device)

    # motion_enc = MotionEncoderBiGRUCo(input_size=opt.dim_movement_latent,
    #                                   hidden_size=opt.dim_motion_hidden,
    #                                   output_size=opt.dim_coemb_hidden,
    #                                   device=opt.device)

    # checkpoint = torch.load(pjoin(opt.checkpoints_dir, opt.dataset_name, 'text_mot_match', 'model', 'finest.tar'),
    #                         map_location=opt.device)
    # movement_enc.load_state_dict(checkpoint['movement_encoder'])
    # text_enc.load_state_dict(checkpoint['text_encoder'])
    # motion_enc.load_state_dict(checkpoint['motion_encoder'])
    # print('Loading Evaluation Model Wrapper (Epoch %d) Completed!!' % (checkpoint['epoch']))
    return textencoder, motionencoder


class EvaluatorModelWrapper(object):

    def __init__(self, opt,device):

        # if opt.dataset_name == 't2m':
        #     opt.dim_pose = 263
        # elif opt.dataset_name == 'kit':
        #     opt.dim_pose = 251
        # else:
        #     raise KeyError('Dataset not Recognized!!!')

        # opt.dim_word = 300
        # opt.max_motion_length = 196
        # opt.dim_pos_ohot = len(POS_enumerator)
        # opt.dim_motion_hidden = 1024
        # opt.max_text_len = 20
        # opt.dim_text_hidden = 512
        # opt.dim_coemb_hidden = 512

        # # print(opt)

        self.text_encoder, self.motion_encoder  = build_models(opt)
        self.opt = opt
        self.device = device

        self.text_encoder.to(self.device)
        self.motion_encoder.to(self.device)
        _data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', 'evaluator')
        self.std = torch.from_numpy(np.load(os.path.join(_data_dir, 'std_can_pos.npy'))).to(self.device)
        self.mean = torch.from_numpy(np.load(os.path.join(_data_dir, 'mean_can_pos.npy'))).to(self.device)
        # self.movement_encoder.to(opt.device)

        self.text_encoder.eval()
        self.motion_encoder.eval()
        # self.movement_encoder.eval()

    # Please note that the results does not following the order of inputs
    def get_co_embeddings(self, texts, motions, m_lens):
        with torch.no_grad():
            et=self.text_encoder(texts).loc
            em = self.get_motion_embeddings(motions,m_lens)
            # word_embs = word_embs.detach().to(self.device).float()
            # pos_ohot = pos_ohot.detach().to(self.device).float()
            # motions = motions.detach().to(self.device).float()
            # # motions = motions - self.mean)

            # '''Movement Encoding'''
            # movements = self.movement_encoder(motions[..., :-4]).detach()
            # m_lens = m_lens // self.opt.unit_length
            # motion_embedding = self.motion_encoder(movements, m_lens)

            # '''Text Encoding'''
            # text_embedding = self.text_encoder(word_embs, pos_ohot, cap_lens)
        return et,em

    # Please note that the results does not following the order of inputs
    def get_motion_embeddings(self, motions, m_lens):
        with torch.no_grad():
            motions = motions - motions[:,:,1:2,0:1]
            motions = motions.detach().to(self.device).float()
            # lengths = [motions.shape[1] for _ in range(motions.shape[0])]
            # mlength_list =lengths
            motions = (motions.reshape(motions.shape[0],motions.shape[1],-1)-self.mean)/self.std
            # align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
            # motions = motions[align_idx]
            # m_lens = m_lens[align_idx]

            # '''Movement Encoding'''
            # movements = self.movement_encoder(motions[..., :-4]).detach()
            # m_lens = m_lens // self.opt.unit_length
            em=self.motion_encoder(motions.float(),list(m_lens.detach().cpu().numpy())).loc
            # motion_embedding = self.motion_encoder(movements, m_lens)
        return em
# pred_motions = pred_motions - pred_motions[:,:,1:2,0:1] ## Canonicalize: Right root to [0,0,0]
# pred_motions = pred_motions.reshape(pred_motions.shape[0],-1,2*21*3)


# # gt_motion =  gt_motion*std_train+ mean_train

# gt_motion = gt_motion - gt_motion[:,:,1:2,0:1]
# gt_motion = gt_motion.reshape(gt_motion.shape[0],-1,2*21*3) ## Canonicalize: Right root to [0,0,0]

# define the 'valid' length, in our setting, all equals to 60, no padding
# lengths = [gt_motion.shape[1] for _ in range(gt_motion.shape[0])]

# ##Encoding after canonicalization
# pred_motions = (pred_motions - mean_enc)/std_enc
# gt_motion = (gt_motion-mean_enc)/std_enc
# mlength_list =lengths