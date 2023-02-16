import torch
import numpy as np
import argparse
import glob
import os
import time
import util
import matplotlib.pyplot as plt
import pandas as pd
from engine import trainer
from itertools import islice

parser = argparse.ArgumentParser()
parser.add_argument('--device',type=str,default='cpu',help='')
parser.add_argument('--data',type=str,default='data/MAX-TEMP',help='data path')
parser.add_argument('--adjdata',type=str,default='data/sensor_graph/adj_mx.pkl',help='adj data path')
parser.add_argument('--adjtype',type=str,default='doubletransition',help='adj type')
parser.add_argument('--aptonly',action='store_true',help='whether only adaptive adj')
parser.add_argument('--addaptadj',action='store_true',help='whether add adaptive adj')
parser.add_argument('--gcn_bool',action='store_true',help='whether to add graph convolution layer')
parser.add_argument('--randomadj',action='store_true',help='whether random initialize adaptive adj')
parser.add_argument('--seq_length',type=int,default=12,help='') # son los y
parser.add_argument('--nhid',type=int,default=32,help='')
parser.add_argument('--splits',type=int,default=10,help='how many splits in the cross validation')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--num_nodes',type=int,default=137,help='number of nodes')
parser.add_argument('--batch_size',type=int,default=64,help='batch size')
parser.add_argument('--learning_rate',type=float,default=0.001,help='learning rate')
parser.add_argument('--dropout',type=float,default=0.3,help='dropout rate')
parser.add_argument('--weight_decay',type=float,default=0.0001,help='weight decay rate')
parser.add_argument('--epochs',type=int,default=20,help='')
parser.add_argument('--from_epochs',type=int,default=0,help='')
parser.add_argument('--print_every',type=int,default=50,help='')
#parser.add_argument('--seed',type=int,default=99,help='random seed')
parser.add_argument('--save',type=str,default='./garage/metr',help='save path')
parser.add_argument('--expid',type=int,default=1,help='experiment id')
parser.add_argument('--no_train', action='store_true',help='use the last saved model')

args = parser.parse_args()




def main():
    os.makedirs(args.save, exist_ok=True)

    device = torch.device(args.device)
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata,args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size, args.splits)
    scaler = dataloader['scaler']
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

    if args.randomadj:
        adjinit = None
    else:
        adjinit = supports[0]

    if args.aptonly:
        supports = None


    
    engine = trainer(scaler, args.in_dim, args.seq_length, args.num_nodes, args.nhid, args.dropout,
                         args.learning_rate, args.weight_decay, device, supports, args.gcn_bool, args.addaptadj,
                         adjinit)


    print("start training...",flush=True)
    his_loss =[]
    val_time = []
    train_time = []
    
    # Codigo nuevo
    train_loss = []
    total_train_loss=[]

    for i in range(args.splits):
        t1 = time.time()
        dataloader[f'train_fold_{i}_loader'].shuffle()
        dataloader[f'test_fold_{i}_loader'].shuffle()
        for iter, (x, y, _, _) in enumerate(dataloader[f'train_fold_{i}_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx= trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy[:,0,:,:])
            train_loss.append(metrics[2])
            total_train_loss.append(metrics[2])
            if iter % args.print_every == 0:
                log = 'Iter: {:03d}, Train Loss: {:.4f}'
                print(log.format(iter, train_loss[-1],flush=True))
        t2 = time.time()
        train_time.append(t2-t1)
        valid_loss = []


        s1 = time.time()
        for iter, (x, y, _, _) in enumerate(dataloader[f'test_fold_{i}_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy[:,0,:,:])
            # preds = engine.model(testx).transpose(1,3)
            # val_outputs.append(preds.squeeze())
            
            valid_loss.append(metrics[2])

        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i,(s2-s1)))
        val_time.append(s2-s1)
        mtrain_loss = np.mean(train_loss)

        mvalid_loss = np.mean(valid_loss)

        his_loss.append(mvalid_rmse)


        log = 'Epoch: {:03d}, Train Loss: {:.4f}, Valid Loss: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_loss, mvalid_loss, (t2 - t1)),flush=True)
        torch.save(engine.model.state_dict(), args.save+"_epoch_"+str(i)+"_"+str(round(mvalid_loss,2))+".pth")

        print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
        print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
        train_loss_file = open("./garage/train_loss.txt", "w")
        val_loss_file = open("./garage/val_loss.txt", "w")
        for element in total_train_loss:
            train_loss_file.write(str(element) + "\n")
        
        for element in his_loss:
            val_loss_file.write(str(element) + "\n")
            

        train_loss_file.close()
        val_loss_file.close()


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2-t1))
