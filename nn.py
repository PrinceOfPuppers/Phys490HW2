import torch
import torch.nn as nn
import torch.nn.functional as funct
import torch.optim as optim
from tqdm import tqdm
class NeralNet(nn.Module):
    def __init__(self,cfg,learningRate):
        print("Creating Network")
        super().__init__()
        self.fcLayers=nn.ModuleList()

        for i in range(0,cfg.fcLayers):
            if i==0:
                inputNodes=cfg.inputLen
            else:
                inputNodes=cfg.fcNodes[i-1]
            
            if i==cfg.fcLayers-1:
                outputNodes=cfg.numLabels
            else:
                outputNodes=cfg.fcNodes[i]

            fcLayer=nn.Linear(inputNodes,outputNodes)
            self.fcLayers.append(fcLayer)
        

        self.device=torch.device("cpu")
        self.to(self.device)

        self.optimizer=optim.SGD(self.parameters(),learningRate)
        self.lossFunct=nn.BCELoss(reduction='mean')
        self.float()
    
    def forward(self,x):

        for i,fcLayer in enumerate(self.fcLayers):
            #last fc layer shouldnt have relu called on it
            if i==len(self.fcLayers)-1:
                x=fcLayer(x)
            else:
                x=funct.relu(fcLayer(x)) 

        return(torch.sigmoid(x))

    def train(self,trainData,trainOneHotVecs,batchSize):
        trainSize=len(trainData)
        right=0
        avgLoss=0
        for i in tqdm(range(0,trainSize,batchSize)):

            imgBatch=trainData[i:i+batchSize].to(self.device)
        
            OneHotVecBatch=trainOneHotVecs[i:i+batchSize].to(self.device)
            self.zero_grad()
            outputs=self(imgBatch)
            loss=self.lossFunct(outputs,OneHotVecBatch)
            loss.backward()
            self.optimizer.step()

            avgLoss+=loss.item()*len(imgBatch)

            #calculates number of right answers
            for j,answerOneHot in enumerate(outputs):
                answer=torch.argmax(answerOneHot)
                correctAnswer=torch.argmax(OneHotVecBatch[j])
                if answer==correctAnswer:
                    right+=1 

        avgLoss=avgLoss/trainSize
        accuracy=right/trainSize
        print("Training Accuracy (decimal):",accuracy,"Training loss:",avgLoss)
        return (accuracy,avgLoss)
    
    def test(self,testData,testOneHotVecs,batchSize):
        right=0
        avgLoss=0
        testSize=len(testData)
        with torch.no_grad():
            for i in tqdm(range(0,testSize,batchSize)):
                imgBatch=testData[i:i+batchSize].to(self.device)
                OneHotVecBatch=testOneHotVecs[i:i+batchSize].to(self.device)
                outputs=self(imgBatch)
                loss=self.lossFunct(outputs,OneHotVecBatch)

                avgLoss+=loss.item()*len(imgBatch)

                #calculates number of right answers
                for j,answerOneHot in enumerate(outputs):
                    answer=torch.argmax(answerOneHot)
                    correctAnswer=torch.argmax(OneHotVecBatch[j])
                    if answer==correctAnswer:
                        right+=1 

        accuracy=right/testSize
        avgLoss=avgLoss/testSize
        print(">>Testing Accuracy (decimal):",accuracy,"Testing Loss:",avgLoss)
        return (accuracy,avgLoss)