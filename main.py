import argparse
import warnings
import os
import random
import numpy as np

from models import *
from layers import *
from loss import DeepMVCLoss
import torch
import scipy.io as sio


from torch.optim import AdamW

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='CLJENet')
parser.add_argument('--load_model', default=False, help='Testing if True or training.')
parser.add_argument('--save_model', default=False, help='Saving the model after training.')

parser.add_argument('--db', type=str, default='MSRCv1',
                    choices=['MSRCv1', 'MNIST-USPS', 'COIL20', 'scene', 'hand', 'Fashion', 'BDGP', 'NUSWIDEOBJ', 'ORL', 'cifar10'],
                    help='dataset name')
parser.add_argument('--seed', type=int, default=10, help='Initializing random seed.')
parser.add_argument("--mse_epochs", default=200, help='Number of epochs to pre-training.')
parser.add_argument("--con_epochs", default=1000, help='Number of epochs to fine-tuning.')
parser.add_argument('-lr', '--learning_rate', type=float, default=0.0001, help='Initializing learning rate.')  # 调整学习率
parser.add_argument('--weight_decay', type=float, default=0., help='Initializing weight decay.')
parser.add_argument("--temperature_l", type=float, default=1.0)
parser.add_argument('--batch_size', default=128, type=int, help='Batch size.')  # 增加批量大小
parser.add_argument('--normalized', type=bool, default=False)
parser.add_argument('--gpu', default='0', type=str, help='GPU device idx.')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":

    if args.db == "MSRCv1":
        args.learning_rate = 0.00001
        args.batch_size = 64
        args.con_epochs = 500
        args.seed = 42
        args.normalized = False
        args.visualize = True  # Enable visualization for this dataset


        dim_high_feature = 4000
        dim_low_feature = 2048
        dims = [512, 1024, 2048]
        lmd = 0.05
        beta = 0.005
        gamma_values = 1

    elif args.db == "BDGP":
        args.learning_rate = 0.0001
        args.batch_size = 64
        args.seed = 42
        args.con_epochs = 500
        args.normalized = True

        dim_high_feature = 500
        dim_low_feature = 1024
        dims = [256, 512]
        lmd = 0.05
        beta = 0.05
        gamma_values = 1


    elif args.db == "Fashion":
        args.learning_rate = 0.0005
        args.batch_size = 100
        args.con_epochs = 500
        args.seed = 20
        args.normalized = True
        args.temperature_l = 0.5

        dim_high_feature = 500
        dim_low_feature = 500
        dims = [256, 512]
        lmd = 0.05
        beta = 0.005
        gamma_values = 1


    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mv_data = MultiviewData(args.db, device)
    num_views = len(mv_data.data_views)
    num_samples = mv_data.labels.size
    num_clusters = np.unique(mv_data.labels).size

    print("==========\nArgs:{}\n==========".format(args))

    print(f"Number of views: {num_views}")
    print(f"Number of samples: {num_samples}")
    print(f"Number of clusters: {num_clusters}")

    input_sizes = np.zeros(num_views, dtype=int)
    for idx in range(num_views):
        input_sizes[idx] = mv_data.data_views[idx].shape[1]

    t = time.time()
    mnw = CLJENetwork(num_views, input_sizes, dims, dim_high_feature, dim_low_feature, num_clusters)
    mnw = mnw.to(device)
    loss_history = []

    gamma = gamma_values
    mvc_loss = DeepMVCLoss(args.batch_size, num_clusters, lambda_=lmd, beta=beta, gamma=gamma)
    optimizer = AdamW(mnw.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    pre_train_loss_values = pre_train(mnw, mv_data, args.batch_size, args.mse_epochs, optimizer)
    t = time.time()
    fine_tuning_loss_values = np.zeros(args.con_epochs, dtype=np.float64)
    for epoch in range(args.con_epochs):
        total_loss = contrastive_train(mnw, mv_data, mvc_loss, args.batch_size, lmd, beta, gamma,
                                           args.temperature_l, args.normalized, epoch, optimizer)

        fine_tuning_loss_values[epoch] = total_loss
        loss_history.append(total_loss / num_samples)


    print("contrastive_train finished.")
    print("Total time elapsed: {:.2f}s".format(time.time() - t))
    if args.save_model:
        torch.save(mnw.state_dict(), f'./models/CLJE_pytorch_model_{args.db}_gamma_{gamma}.pth')

    acc, nmi, pur, ari = valid(mnw, mv_data, args.batch_size)
    with open(f'result_{args.db}_gamma_{gamma}.txt', 'a+') as f:
        f.write('{} \t {} \t {} \t {} \t {} \t {} \t {} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.6f} \t {:.4f} \n'.format(
            dim_high_feature, dim_low_feature, args.seed, args.batch_size,
            args.learning_rate, lmd, beta, gamma, acc, nmi, pur, ari, (time.time() - t)))
        f.flush()