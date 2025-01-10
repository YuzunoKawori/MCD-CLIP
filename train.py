import os
import torch 
import numpy as np
import torch.utils
from skimage.util import view_as_windows
from torch.utils.data import random_split, DataLoader
import random
import wandb
import time
import argparse
import torch
from torchmetrics import AUROC
from sklearn.metrics import confusion_matrix
import matplotlib
from dataset import CheXpertDataset
matplotlib.use('Agg')
import model_mcd
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
torch.set_printoptions(threshold=np.inf)
class_x=['normal lungs','abnormal lungs']
task_names = [ 'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 
              'Lung Lesion','Lung Opacity','Pleural Effusion', 'Pleural Other', 'Pneumonia', 
             'Pneumothorax', 'Support Devices']
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
def train(train_loader,model,arg,fix,val_loader,train_set):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sum_loss=0
    train_correct=0
    optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=fix, weight_decay=args.reg)
    criterion1 = torch.nn.CrossEntropyLoss()
    max_auc=0
    for epoch in range(arg.epochs):
        sum_loss=0
        train_correct=0
        batch_num=1
        if len(train_set)%arg.batch_size==0:
            all=len(train_set)//arg.batch_size
        else:
            all=(len(train_set)//arg.batch_size)+1
        model.train()
        for i,data in enumerate(train_loader,0):
            print("epoch:{:}, {:}/{:}".format(epoch+1,batch_num,all))
            images = data[:-1] 
            y = data[-1]  
            frontal_images = images[0]
            lateral_images = images[1] 
            x= torch.stack((frontal_images, lateral_images), dim=1)
            x = x.repeat(1, 1, 3, 1, 1) 
            x=x.to(device)
            y=y.to(device)
            y=y.long()
            x=x.to(torch.float32)
            kekka=model(x)
            loss = criterion1(kekka, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _,id=torch.max(kekka.data,1)
            sum_loss+=loss.data
            train_correct+=torch.sum(id==y.data)
            batch_num+=1
        wandb.log({
        "train_loss": sum_loss / len(train_loader)})
        wandb.log({
        "train_acc":  (100 * train_correct / len(train_set))})
        print('[%d,%d] loss:%.03f' % (epoch + 1, arg.epochs, sum_loss / len(train_loader)))
        print('        correct:%.03f%%' % (100  *train_correct / len(train_set)))
        print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        val_loss = 0
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        val_correct=0
        val_auc_en = AUROC(task='binary', average="macro")  # Accuracy
        pre_val=[]
        true_val=[]
        pre=[]
        auc=0
        num=0
        with torch.no_grad():
            for i,data in enumerate(val_loader,0):
                images = data[:-1]  
                y = data[-1] 
                frontal_images = images[0]
                lateral_images = images[1] 
                x= torch.stack((frontal_images, lateral_images), dim=1)
                x = x.repeat(1, 1, 3, 1, 1) 
                x.to(device)
                y.to(device)
                y=y.long()
                x=x.to(torch.float32)
                kekka=model(x)
                kekka = kekka[:, :2].to(device)
                y=y.to(device)
                loss = criterion1(kekka, y)
                pre.append(kekka)
                kekka=kekka.softmax(dim=-1)
                val_loss+=loss.data
                _,id=torch.max(kekka.data,1)
                pre_val.append(id)
                true_val.append(y.data)
                val_correct+=torch.sum(id==y.data)
                num+=1
            pre= torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
            pre_val=torch.tensor(pre_val)
            true_val=torch.tensor(true_val)
            pre=torch.squeeze(pre,dim=1).to(device)
            pre=pre.softmax(dim=-1)
            pre= pre[:,1]
            val_auc=val_auc_en(pre.cpu(),true_val)
            print('[%d,%d] val_loss:%.03f' % (epoch + 1, arg.epochs, val_loss / num))
            wandb.log({
            "val_loss": val_loss / num})
            wandb.log({
                "Val Auc": val_auc})
            print(val_auc)
            print(max_auc)
            if  val_auc>max_auc:
                max_auc=val_auc
                print("save model")
                torch.save(model.state_dict(),"model.pth")
def test(test_loader,arg,testset):
    test_correct=0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.MCDCLIP(class_x,20,768,512,512,512)
    model = model.to(torch.float32)
    model.load_state_dict(torch.load('model.pth'))
    model.to(device)
    model.eval()
    test_auc_en = AUROC(task='binary', average="macro") 
    pre=[]
    pre_test=[]
    true_test=[]
    num=0
    with torch.no_grad():
        for i,data in enumerate(test_loader,0):
            images = data[:-1] 
            y = data[-1]  
            if y==2:
                continue
            frontal_images = images[0]  
            lateral_images = images[1]  
            x= torch.stack((frontal_images, lateral_images), dim=1)
            x = x.repeat(1, 1, 3, 1, 1) 
            x.to(device)
            y.to(device)
            y=y.long()
            x=x.to(torch.float32)
            kekka=model(x)
            kekka = kekka[:, :2].to(device)
            y=y.to(device)
            pre.append(kekka)
            kekka=kekka.softmax(dim=-1)
            values, indices = kekka[0].topk(2)
            print("\nTop predictions:\n")
            for value, index in zip(values, indices):
                print(f"{class_x[index]:>16s}: {100 * value.item():.2f}%")
            _,id=torch.max(kekka.data,1)
            pre_test.append(id)
            true_test.append(y.data)
            test_correct+=torch.sum(id==y.data)
            print("pre:")
            print(id)
            print("true")
            print(y.data)
            num+=1
        pre= torch.tensor([item.cpu().detach().numpy() for item in pre]).cuda()
        pre_test=torch.tensor(pre_test)
        true_test=torch.tensor(true_test)
        pre=torch.squeeze(pre,dim=1).to(device)
        pre=pre.softmax(dim=-1)
        pre= pre[:,1]
        test_auc=test_auc_en(pre.cpu(),true_test)
        print(test_auc)
        pre_test=torch.tensor(pre_test)
        true_test=torch.tensor(true_test)
        print("Accuracy:{:.03f}%".format(100 *test_correct/num))
        wandb.log({
            "Auc": test_auc})
        wandb.log({
            "Acc": (100 *test_correct/len(testset))})
    return test_correct/len(testset)
run = wandb.init(project="Abain_new",
                 entity="yuzunokawori"
                 )     
parser = argparse.ArgumentParser(description='PyTorch MCDCLIP bags Example')
parser.add_argument('--epochs', type=int, default=60, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--reg', type=float, default=10e-5, metavar='R',
                    help='weight decay')
parser.add_argument('--seed', type=int, default=80, metavar='S',
                    help='random seed (default: 3)')
parser.add_argument('--device', type=int, default=0, metavar='D',
                    help='gpu (default: 0)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

parser.add_argument('--batch_size', type=int, default=64,help='batch_size')
parser.add_argument('--label_rate', type=int, default=0.6,help='label_rate')
parser.add_argument('--namda',type=float,default=[0.5,0.5])
parser.add_argument('--num_view',type=int,default=2,help="num of view")
parser.add_argument('--num_for_train',type=int,default=2,help="num for train the model")
parser.add_argument('--choose_few_shot',type=int,default=0,help="if use few_shot train")
parser.add_argument('--num_class',type=int,default=2,help="num of class")
parser.add_argument('--dropout_rate',type=int,default=0.5,help="the rate of dropout")

args = parser.parse_args()
print(args)
wandb.config.update(args)
print(torch.cuda.device_count())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(args.device)
print('GPU: ', torch.cuda.current_device())
args.lr=wandb.config.lr
args.batch_size=wandb.config.batch_size
args.reg=wandb.config.reg
print(wandb.config)
setup_seed(args.seed)

for class_name in task_names:
    print(class_name)
    train_dataset = CheXpertDataset('/home/songyue/.workplace/data/data/chexpert-small-csy-custom-train.h5', task=class_name)
    val_dataset = CheXpertDataset('/home/songyue/.workplace/data/data/chexpert-small-csy-custom-val.h5', task=class_name)
    test_dataset = CheXpertDataset('/home/songyue/.workplace/data/data/chexpert-small-csy-custom-test.h5', task=class_name)
    print(f"the number of train set: {len(train_dataset)}")
    print(f"the number of validation set: {len(val_dataset)}")
    print(f"the number of test set: {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,num_workers=8,
                                pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    model = model.MCDCLIP(class_x,20,768,512,512,512)
    model = model.to(torch.float32)
    model.to(device)
    wandb.watch(model)
    print("Train start: -----------------------------------------------------------")
    train(train_loader,model,args,args.lr,val_loader,train_dataset)
    print("Test start: -----------------------------------------------------------")
    now_acc=test(test_loader,args,test_dataset)
