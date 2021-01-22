# > IMPORTS

# Libraries

from pathlib import Path

# Spyrit
import sys
import random
import argparse
sys.path.append('../../spyrit/')

from spyrit.learning.model_Had_DCAN import *
from spyrit.learning.nets import *

if __name__ == "__main__":

    img_size= 64
    data_root="../../data/"
    nbr_to_disp = 5
    model_root= "models/SDCAN/"
    test_string="Marc et Juliette"
    batch_size = 256
    i_test= 7
    precompute_root="models/SDCAN/"
    M = 333                        # nombre de coeff d'Adamar pris en compte pour traiter l'image
    net_arch = 0 

    transform =transforms.Compose(
        [transforms.CenterCrop(img_size*2),      # rogne les côté l'image d'entrée pour avoir une image de taille 128*128
         transforms.functional.to_grayscale,     # transforme l'image en niveau de gris
         transforms.ToTensor(),                  # transforme l'image (PIL) en tensor
         transforms.Normalize([0.5], [0.5]),     # normalise le tensor autour de 0.5 0.5
         transforms.FiveCrop(img_size),          # transforme le tensor en une liste de 5 tensors. Chaque tensor est une image de
                                                 # taille 64*64 prise aux quatre coins et au centre de l'image d'entrée
        ])

    # download the existing average and cov files
    my_average_file = Path(precompute_root) / ('Average_{}x{}'.format(img_size, img_size)+'.npy')
    my_cov_file = Path(precompute_root) / ('Cov_{}x{}'.format(img_size, img_size)+'.npy')

    #list to save tensors
    bd = []
    hd = []
    hg = []
    bg = []
    rechg = []
    rechd = []
    recbd = []
    recbg = []
    recc= []

    # Torch Init
    device = torch.device('cpu')
    torch.manual_seed(7)

    # Dataset Loader
    testset = \
        torchvision.datasets.STL10(root=data_root, split='test', download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    dataloaders = {'val': testloader}

    # Network Init
    if not(my_average_file.is_file()) or not(my_cov_file.is_file()):
        print('Can\'t find the file')
    else:
        print('Loading covariance and mean')
        Mean_had = np.load(my_average_file)
        Cov_had  = np.load(my_cov_file)

    model = compNet(img_size, M, Mean_had, Cov_had, net_arch)

    # Network Load
    filename = model_root + 'NET_c0mp_N_64_M_333_epo_5_lr_0.001_sss_10_sdr_0.5_bs_1024_reg_1e-07'
    model = model.to(device)
    load_net(filename, model)

    print('Evaluation de l\'image n° :')
    for i in range(nbr_to_disp):
        inputs, labels = next(iter(dataloaders['val']))
        inputsHG = inputs[0]    #save top left tensor of the 128*128 image
        inputsHD = inputs[1]    #save top rigth tensor of the 128*128 image
        inputsBG = inputs[2]    #save botowm left tensor of the 128*128 image
        inputsBD = inputs[3]    #save botowm rigth tensor of the 128*128 image
        inputsC = inputs[4]     #save center tensor of the 128*128 image
        inputsHG = inputsHG.to(device)
        inputsBD = inputsBD.to(device)
        inputsHD = inputsHD.to(device)
        inputsBG = inputsBG.to(device)
        inputsC = inputsC.to(device)
        bd.append(inputsBD[i_test + i, 0, :, :].cpu().detach().numpy())
        hd.append(inputsHD[i_test + i, 0, :, :].cpu().detach().numpy())
        bg.append(inputsBG[i_test + i, 0, :, :].cpu().detach().numpy())
        hg.append(inputsHG[i_test + i, 0, :, :].cpu().detach().numpy())
        #evaluate the tensor in the model
        recbd.append(model.evaluate(inputsBD[(i_test + i):(i_test + i + 1), :, :, :]).cpu().detach().numpy().squeeze())
        rechd.append(model.evaluate(inputsHD[(i_test + i):(i_test + i + 1), :, :, :]).cpu().detach().numpy().squeeze())
        recbg.append(model.evaluate(inputsBG[(i_test + i):(i_test + i + 1), :, :, :]).cpu().detach().numpy().squeeze())
        rechg.append(model.evaluate(inputsHG[(i_test + i):(i_test + i + 1), :, :, :]).cpu().detach().numpy().squeeze())
        recc.append(model.evaluate(inputsC[(i_test + i):(i_test + i + 1), :, :, :]).cpu().detach().numpy().squeeze())
        print(i + 1, '/', nbr_to_disp, ' ')

        # Plot nbr_to_disp number of image
    for i in range(nbr_to_disp):

        #concatenate the top left and the top rigth evaluate tensor
        rech = np.concatenate((rechg[i], rechd[i]), axis=1)
        #concatenate the bottom left and the bottom rigth evaluate tensor
        recb = np.concatenate((recbg[i], recbd[i]), axis=1)
        #concatenate the top and the bottom evaluate tensor
        recall = np.concatenate((rech, recb))

        #same for the non evaluate tensor
        ch = np.concatenate((hg[i], hd[i]), axis=1)
        cb = np.concatenate((bg[i], bd[i]), axis=1)
        all = np.concatenate((ch, cb))

        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        fig.suptitle('c0mp' + ' -- Test: ' + test_string, fontsize=20)

        ax = axs[0]
        im = ax.imshow(all)
        ax.set_title("Ground-truth-reconstruction-bloc")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)
        fig.colorbar(im, ax=ax)

        ax = axs[1]
        im = ax.imshow(recall)
        ax.set_title("Train-reconstruction-bloc")
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.grid(False)
        fig.colorbar(im, ax=ax)

        plt.show()
