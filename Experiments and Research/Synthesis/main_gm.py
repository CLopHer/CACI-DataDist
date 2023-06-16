import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, evaluate_synset, get_daparam, match_loss, get_time, TensorDataset, epoch


def main():

    parser = argparse.ArgumentParser(description='Parameter Processing')
    # number of image for each class
    image_per_class = 1
    # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer
    eval_mode = 'S' 
    # the number of experiments
    num_exp = 1
    # the number of evaluating randomly initialized models
    num_eval = 5 
    # epochs to train a model with synthetic data
    epoch_eval_train = 20
    # traing iteration
    train_iteration = 5
    # learning rate for updating synthetic images
    lr_img = 0.1
    # learning rate for updating network parameters
    lr_net = 0.01
    # batch size for test data
    batch_test = 64
    # batch size for training networks
    batch_train = 64
    # initialization of synthetic data, noise/test: initialize from random noise or test images. The two initializations will get similar performances.
    syn_init = 'noise'
    # distance metric
    dis_metric = 'ours'
    # For speeding up, we can decrease the Iteration and epoch_eval_train, which will not cause significant performance decrease.


    args = parser.parse_args()
    outer_loop, inner_loop = get_loops(image_per_class)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists('data'):
        os.mkdir('data')

    if not os.path.exists('result'):
        os.mkdir('result')

    eval_it_pool = np.arange(0, train_iteration+1, 50).tolist() if eval_mode == 'S' else [train_iteration] # The list of iterations when we evaluate models and record results.
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset('data')
    model_eval_pool = ['ResNet34']


    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []


    for exp in range(num_exp):
        print('\n================== Exp %d ==================\n '%exp)
        print('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the test dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0).to(device)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=device)

        for c in range(num_classes):
            print('class c = %d: %d test images'%(c, len(indices_class[c])))

        def get_images(c, n): # get random n images from class c
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle]

        for ch in range(channel):
            print('test images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes*image_per_class, channel, im_size[0], im_size[1]), dtype=torch.float, requires_grad=True, device=device)
        label_syn = torch.tensor([np.ones(image_per_class)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

        if syn_init == 'test':
            print('initialize synthetic data from random test images')
            for c in range(num_classes):
                image_syn.data[c*image_per_class:(c+1)*image_per_class] = get_images(c, image_per_class).detach().data
        else:
            print('initialize synthetic data from random noise')


        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=lr_img, momentum=0.5) # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(device)
        print('%s training begins'%get_time())

        for it in range(train_iteration+1):

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%('ResNet18', model_eval, it))
                    param_augment = get_daparam('FashionMNIST', model_eval)
                    accs = []
                    for it_eval in range(num_eval):
                        net_eval = get_network(channel, num_classes).to(device) # get a random model
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach()) # avoid any unaware modification
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, lr_net, batch_train, param_augment, device, epoch_eval_train)
                        accs.append(acc_test)
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

                    if it == train_iteration: # record the final results
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join('result', 'vis_%s_%s_%dipc_exp%d_iter%d.png'%('FashionMNIST', 'ResNet18', image_per_class, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch]  * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis<0] = 0.0
                image_syn_vis[image_syn_vis>1] = 1.0
                save_image(image_syn_vis, save_name, nrow=image_per_class) # Trying normalize = True/False may get better visual effects.
                # The generated images would be slightly different from the visualization results in the paper, because of the initialization and normalization of pixels.


            ''' Train synthetic data '''
            net = get_network(channel, num_classes).to(device) # get a random model
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=lr_net, momentum=0.5)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0


            for ol in range(outer_loop):

                ''' freeze the running mu and sigma for BatchNorm layers '''
                # Synthetic data batch, e.g. only 1 image/batch, is too small to obtain stable mu and sigma.
                # So, we calculate and freeze mu and sigma for BatchNorm layer with test data batch ahead.
                # This would make the training with BatchNorm layers easier.

                BN_flag = False
                BNSizePC = 16  # for batch normalization
                for module in net.modules():
                    if 'BatchNorm' in module._get_name(): #BatchNorm
                        BN_flag = True
                if BN_flag:
                    breakpoint()
                    img_test = torch.cat([get_images(c, BNSizePC) for c in range(num_classes)], dim=0)
                    net.train() # for updating the mu, sigma of BatchNorm
                    output_test = net(img_test) # get running mu, sigma
                    for module in net.modules():
                        if 'BatchNorm' in module._get_name():  #BatchNorm
                            module.eval() # fix mu and sigma of every BatchNorm layer


                ''' update synthetic data '''
                loss = torch.tensor(0.0).to(device)
                for c in range(num_classes):
                    img_test = get_images(c, batch_test)
                    lab_test = torch.ones((img_test.shape[0],), device=device, dtype=torch.long) * c
                    output_test = net(img_test)
                    loss_test = criterion(output_test, lab_test)
                    gw_test = torch.autograd.grad(loss_test, net_parameters)
                    gw_test = list((_.detach().clone() for _ in gw_test))

                    img_syn = image_syn[c*image_per_class:(c+1)*image_per_class].reshape((image_per_class, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((image_per_class,), device=device, dtype=torch.long) * c
                    output_syn = net(img_syn)
                    loss_syn = criterion(output_syn, lab_syn)
                    gw_syn = torch.autograd.grad(loss_syn, net_parameters, create_graph=True)

                    loss += match_loss(gw_syn, gw_test, device, dis_metric)

                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()
                loss_avg += loss.item()

                if ol == outer_loop - 1:
                    break


                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=64, shuffle=True, num_workers=0)
                for il in range(inner_loop):
                    epoch('train', trainloader, net, optimizer_net, criterion, None, device)


            loss_avg /= (num_classes*outer_loop)

            if it%10 == 0:
                print('%s iter = %04d, loss = %.4f' % (get_time(), it, loss_avg))

            if it == train_iteration: # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join('result', 'res_%s_%s_%dipc.pt'%('FashionMNIST', 'ResNet18', image_per_class)))


    print('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        print('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%'%(num_exp, 'ResNet18', len(accs), key, np.mean(accs)*100, np.std(accs)*100))



if __name__ == '__main__':
    main()