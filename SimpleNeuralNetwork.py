from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
import numpy as np
import torch
import cv2
import os 

class SimpleNeuralNetwork(torch.nn.Module):
    def __init__(self,num,input_size,Neuron):
        super(SimpleNeuralNetwork,self).__init__()
        self.input_size = input_size
        self.fc1 = torch.nn.Linear(self.input_size*self.input_size,Neuron)
        self.fc2 = torch.nn.Linear(Neuron,num)

    def forward(self,x):
        x = x.view(-1,self.input_size*self.input_size)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)


class AI:
    def __init__(
        self,
        EPOCH       =   40,
        IMAGE_SIZE  =   160,
        HIDDEN_1    =   320,
        LR          =   0.00005,
        model_num   =   3,
        TRAIN_DIR   =   "selected-dataset/train/",
        TEST_DIR    =   "selected-dataset/test/",
        PT_NAME     =   "simplenet.pt",
        LOSS_PNG    =   "simplenet-loss.png",
        ACC_PNG     =   "simplenet-acc.png"
        ):
        self.EPOCH      =   EPOCH
        self.IMAGE_SIZE =   IMAGE_SIZE
        self.HIDDEN_1   =   HIDDEN_1
        self.LR         =   LR 
        self.MODEL      =   SimpleNeuralNetwork(num=20,input_size=IMAGE_SIZE,Neuron=HIDDEN_1)
        self.OPTIMIZER  =   torch.optim.Adam(params=self.MODEL.parameters(),lr=self.LR)
        self.TRAIN_DIR  =   TRAIN_DIR
        # self.TEST_DIR   =   TEST_DIR
        self.PT_NAME    =   PT_NAME
        self.LOSS_PNG   =   LOSS_PNG
        self.ACC_PNG    =   ACC_PNG

        ######------学習用データローダー-----#####
        train_data  =   torchvision.datasets.ImageFolder(
            root=self.TRAIN_DIR,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.Grayscale(),
                torchvision.transforms.Resize((self.IMAGE_SIZE,self.IMAGE_SIZE)),
                torchvision.transforms.ToTensor(),
            ])
        )
        self.train_data =   torch.utils.data.DataLoader(
            train_data,
            batch_size=1,
            shuffle=True
        )

    def train(self):
        for data in tqdm(self.train_data):
            x,target    =   data 
            self.OPTIMIZER.zero_grad()
            output      =   self.MODEL(x)
            ######------損失関数-----#####
            loss = F.nll_loss(output,target)
            loss.backward()
            self.OPTIMIZER.step()
        return loss 

    def test(self,test_dir,name,label):
        self.TEST_DIR   =   test_dir
        #   学習停止
        self.MODEL.eval()

        total   =   0
        correct =   0
        with torch.no_grad():
            for f in tqdm(os.listdir(self.TEST_DIR)):
                #   画像を読み込み
                img     =   cv2.imread(self.TEST_DIR+"/"+f)
                imgGray =   cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img     =   cv2.resize(imgGray,(self.IMAGE_SIZE,self.IMAGE_SIZE))
                img = np.reshape(img,(1,self.IMAGE_SIZE,self.IMAGE_SIZE))
                img = np.transpose(img,(1,2,0))
                img = img.astype(np.uint8)
                mInput = torchvision.transforms.ToTensor()(img) 
                mInput = mInput.view(-1, self.IMAGE_SIZE*self.IMAGE_SIZE)
                output = self.MODEL(mInput)
                p = self.MODEL.forward(mInput)
                if p.argmax() == label:
                    correct += 1
                total   +=  1
        percent =   100*correct/total
        print("{}{:>10f}".format(name,percent))
        return 

    def save_loss_png(self,loss):
        plt.figure()
        plt.plot(range(1,self.EPOCH+1),loss,label="trainLoss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(self.LOSS_PNG)
    
    def save_acc_png(self,acc,name):
        plt.figure()
        plt.plot(range(1,self.EPOCH+1),acc,label=str(name))
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.savefig(str(name)+"_"+self.ACC_PNG)

    def save_model(self):
        torch.save(self.MODEL.state_dict(),self.PT_NAME)



class AI_30Classes:
    def __init__(self,root_train_dir,root_test_dir):
        ai  =   AI(TRAIN_DIR=root_train_dir)

        animal_classes  =   [
            [
                [],[],[]
            ]  for i in os.listdir(root_test_dir)
        ]

        loss    =   []
        for e in range(ai.EPOCH):
            loss.append(ai.train())
            for i,name in enumerate(os.listdir(root_test_dir)):
                test_path               =   root_test_dir+name+"/"
                animal_classes[i][0]    =   name 
                animal_classes[i][1]    =   i
                animal_classes[i][2].append(ai.test(test_path,name,i))
        

        ai.save_loss_png(loss)

        for x in animal_classes:
            # print(x)
            ai.save_acc_png(x[2],x[0]) 

        #   モデルの保存
        ai.save_model()       


if __name__ == "__main__":
    ai_30classes    =   AI_30Classes(
        root_train_dir="selected-dataset/train/",
        root_test_dir="selected-dataset/test/"
    )