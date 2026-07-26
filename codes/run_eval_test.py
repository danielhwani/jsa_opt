import os
from config import config
import torch
from torch.utils.data import DataLoader
import dataset
import model.model_joint_sa as model_joint_sa
import eval
import utils.utils_options as option

test_set = dataset.Dataset(config, train=False)
test_loader = DataLoader(dataset=test_set, batch_size=1, num_workers=2)

net = model_joint_sa.JSA_transformer(
    img_size=config['patch_size'], embedded_dim=config['embed_dim'],
    win_size=8, projection_option='linear', ffn_option='mlp',
    depths=[1,2,4,8,2,8,4,2,4], in_x=config['x_dim'], in_f=config['f_dim'])
net = net.to('cuda:0')

checkpoint_dir = os.path.join(config["data_dir"], config["task"], "__checkpoints__")
option.load_checkpoint(config['task'], checkpoint_dir, net, 'best')

eval.eval_test(net, test_loader, 'best')
