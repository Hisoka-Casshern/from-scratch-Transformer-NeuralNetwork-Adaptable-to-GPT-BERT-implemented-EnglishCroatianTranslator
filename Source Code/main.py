from rawDatasetCreation import DesiredLangugeRawDataset
from trainingScript import train
import os
from pathlib import Path
import torch

if __name__ == '__main__':
    #switch to current dir
    os.chdir(Path(__file__).parent)
    #create the database
    if(os.path.isdir('EnglishCroatianDatabase') == False):
        DesiredLangugeRawDataset().generateDatasetFromListOfDatasets()
    #train model
    train(pathToDataSet='EnglishCroatianDatabase/en_hr_translationsDatabase',
          splitTrain=0.9,
          sequenceLength=180,
          batchSize=32,
          sourceLanguge='english',
          targetLanguge='croatian',
          modelEmbeddingDimension=512,
          numberOfAttentionHeads=8,
          numberOfEncoderBlocks=6,
          numberOfDecoderBlocks=6,
          dimensionExpansionFactorInFeedForward=4,
          dropout=0.05,
          optimizer=torch.optim.Adam,
          learningRate=0.0001,
          lossFunction=torch.nn.CrossEntropyLoss,
          numberOfEpochs=5,
          continueTrain=False,
          deviceStr='cuda',
          numberOfWorkers = 12,
          learningRateDecay = False)
