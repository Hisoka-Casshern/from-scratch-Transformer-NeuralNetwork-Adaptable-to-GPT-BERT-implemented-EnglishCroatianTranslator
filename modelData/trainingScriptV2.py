from googletrans import LANGUAGES
from datasetCreationScript import DatasetEx
from transformerScript import TransformerModel
import torch
import torch.utils.data as torchUtils
import os
import tokenizers
import numpy as np
import pandas as pd
import sys

def trainGeneratingTokenByToken(
                      model:TransformerModel,
                      inputToEncoderData:torch.tensor,inputMask:torch.tensor,
                      deviceStr:str='cuda',maximumSequenceLength:int=180,
                      targetTokenizer:tokenizers.Tokenizer = None,
                      targetVocabSize=None):
    '''I created this function to train the network every other epoch such that it also
    is aware of token by token, meaning using this the transforemer relies only
    on input to encoder and labels to correct itself, I do this as the model
    will be used token by token generation,and every other code i have seen is sketchy
    about their results and in my opinion are not shopwing the full results of their networks
    that are probably just bad and they oversell it, this is my aproach to corectly implement
    as many public codes are not fully implemented as they say they are.'''
    #device
    device = torch.device(deviceStr)
    
    #now as in normal token by token we input just put [SOS] token into decoder inputs as we
    #want it to start generate ontop of start token
    #get target [SOS] and [EOS] tokens but also [PAD] token from tokenizer inisde modelData directory 
    targetSOSid = targetTokenizer.token_to_id('[SOS]')
    targetEOSid = targetTokenizer.token_to_id('[EOS]')
    targetPADid = targetTokenizer.token_to_id('[PAD]')

    #now make a buffer to store each generated output
    outputs=[]
    #here we create the same number of inputs as the size of a training batch
    #create the inputs with [SOS] and account for batch dimension when broadcasting 
    decoderInputs = \
        [(torch.ones(1,1,dtype=inputToEncoderData.dtype)*targetSOSid).to(device) 
                                            for _ in range(inputToEncoderData.shape[0])]
    #now we for each embeded sentence generate outputs
    #the input innto the necoder needs to be also just one sentence taken from encoderInputData along the batch 
    #dimension 
    for idx in range(len(decoderInputs)):
        #inputs to encoder
        currentInputToEncoder = inputToEncoderData[idx]
        currentEncoderMask = inputMask[idx]
        #get encoder output
        encoderOutput = model.transformerEncoder(inputToEncoder=currentInputToEncoder,
                                                 inputMask=currentEncoderMask)
        #so now take sentence for decoder that will be generated token by token
        currentInputToDecoder = decoderInputs[idx]
        #we create lower triangular mask for token prediction so that future tokens dont have influence
        decoderMask = torch.tril(torch.ones(1,currentInputToDecoder.size(1),currentInputToDecoder.size(1),
                                            dtype=inputToEncoderData.dtype))
        decoderMask = decoderMask.to(device)
        #get the decoder output here and using decoder mask
        decoderOutput = model.transformerDecoder(inputToDecoder=currentInputToDecoder,
                                                targetMask=decoderMask,
                                                outputFromEncoder=encoderOutput,
                                                inputMask=currentEncoderMask)
        #we create the input of [SOS] and account for batch dimension when broadcasting 
        decoderInput = torch.ones(1,1,dtype=inputToEncoderData.dtype)*targetSOSid
        decoderInput = decoderInput.to(device)
        #no we run the decoder and generate tokens as usual, only diference is in case we reach EOS
        while(True):
            #check if we reached the sequence length, shape(batchSize,sequnceLength)
            if(decoderInput.size(1) == maximumSequenceLength):
                #here append the output
                outputs.append(decoderInput)
                break
            #we create lower triangular mask for token prediction so that future tokens dont have influence
            decoderMask = \
                torch.tril(torch.ones(1,decoderInput.size(1),decoderInput.size(1),dtype=inputToEncoderData.dtype))
            decoderMask = decoderMask.to(device)
            #now we generate next token using projection ,projection take shape(batch, seq_len, d_model) 
            ##we care about only the projection of the last token and 
            #in case of last token its shape is (batch, d_model)
            #and then projectionn gives shape(batch, vocab_size)
            projectionOutput = model.transformerProjection(outputFromDecoder=decoderOutput[:,-1])       
            #now to get the next word we take the one with maximum probabilty, meaning along the dictionary
            nextWord = torch.max(projectionOutput,dim=1)[1]
            #we add it to the decoderInput (columns as this is sequence dimension) and check if it is [EOS]
            nextWord = torch.tensor([[nextWord.item()]],dtype=inputToEncoderData.dtype)
            nextWord = nextWord.to(device)
            decoderInput=torch.cat([decoderInput,nextWord],dim=1)
            if(nextWord.item() == targetEOSid):
                #here in the case we reach the EOS, we PAD the rest of sentence and exit
                numberOfPADtokens = maximumSequenceLength - decoderInput.shape[1]
                if(numberOfPADtokens>0):
                    PADtokensForTarget = \
                        torch.tensor([targetPADid ]*numberOfPADtokens).to(device)
                    PADtokensForTarget = PADtokensForTarget[None,:]
                    decoderInput = torch.cat([decoderInput,PADtokensForTarget],dim=1)   
                #here append the output
                outputs.append(decoderInput)
                break

    #now for each output we make a tensor such that we put 1 to correspoding position in a 
    #vocabulary and zero elsewhere, meaning we use tokens as indeces in vocabulary
    expandedOutputsToVocabulary = []
    for outputTensor in outputs:
        #create a tensor of zeros of shape(sequenceLength, vocabularySize)
        currentTensor = torch.zeros(maximumSequenceLength,targetVocabSize)
        #now fill with ones
        row = 0
        for indexColumnPosition in list(outputTensor.detach().cpu().numpy())[0]:
            currentTensor[row,indexColumnPosition] = 1
            row+=1
        #put to gpu and put inside buffer, also we account for batch dimension
        currentTensor = currentTensor[None,:,:]
        currentTensor = currentTensor.to(device)
        expandedOutputsToVocabulary.append(currentTensor)

    #now conacatenate all outputs into one batched output
    output = torch.cat(expandedOutputsToVocabulary,dim=0)

    #return output shaped (batchSize, sequenceLength,targetVocabularySize) 
    return output


def generateTokenByToken(model:TransformerModel,
                        inputData:torch.tensor,inputMask:torch.tensor,
                        deviceStr:str='cuda',maximumSequenceLength:int=180,
                        targetTokenizer:tokenizers.Tokenizer = None):
    '''here we dont translate all at once but token by token, so first we need the encoder
    output and then we get each next token from decoder until we reach [EOS] index or
    sequencle maximum length that is 180 in our case, this vaires based on size of model
    and input data format sentences length that are then tokenized'''
    #device
    device = torch.device(deviceStr)
    #get encoder output
    encoderOutput = model.transformerEncoder(inputToEncoder=inputData,inputMask=inputMask)
    
    #now we input just put [SOS] token into decoder inputs as we want it to start generate ontop of 
    #start token
    #first get target [SOS] and [EOS] tokens from tokenizer inisde modelData directory 
    targetSOSid = targetTokenizer.token_to_id('[SOS]')
    targetEOSid = targetTokenizer.token_to_id('[EOS]')
    #we create the input of [SOS] andaccount for batch dimension when broadcasting 
    decoderInput = torch.ones(1,1,dtype=inputData.dtype)*targetSOSid
    decoderInput = decoderInput.to(device)
    #no we run the decoder and generate tokens
    endOfGeneration = False
    while(endOfGeneration == False):
        #check if we reached the sequence length, shape(batchSize,sequnceLength)
        if(decoderInput.size(1) == maximumSequenceLength):
            endOfGeneration = True
            continue
        #we create lower triangular mask for token prediction so that future tokens dont have influence
        decoderMask = \
            torch.tril(torch.ones(1,decoderInput.size(1),decoderInput.size(1),dtype=inputData.dtype))
        decoderMask = decoderMask.to(device)
        #get the decoder output
        decoderOutput = model.transformerDecoder(inputToDecoder=decoderInput,targetMask=decoderMask,
                                                 outputFromEncoder=encoderOutput, inputMask=inputMask)

        #now we generate next token using projection ,projection take shape(batch, seq_len, d_model) 
        ##we care about only the projection of the last token and 
        #in case of last token its shape is (batch, d_model)
        #and then projectionn gives shape(batch, vocab_size)
        projectionOutput = model.transformerProjection(outputFromDecoder=decoderOutput[:,-1])       
        #now to get the next word we take the one with maximum probabilty, meaning along the dictionary
        nextWord = torch.max(projectionOutput,dim=1)[1]
        #we add it to the decoderInput (columns as this is sequence dimension) and check if it is [EOS]
        nextWord = torch.tensor([[nextWord.item()]],dtype=inputData.dtype)
        nextWord = nextWord.to(device)
        decoderInput=torch.cat([decoderInput,nextWord],dim=1)
        if(nextWord.item() == targetEOSid):
            endOfGeneration = True
            continue

    #return
    return decoderInput[0]


def modelValidationTokenByToken(model:TransformerModel,
                                validationDataSet:torch.tensor,maxSequnceLength:int,
                                deviceStr:str = 'cuda',
                                targetTokenizer:tokenizers.Tokenizer = None):
    '''here we run the validation of our transformer over its dataset,
    by generating senteces token by token'''
    #make buffer to store sentences
    inputText =[]
    targetText =[]
    predictedText =[]
    #switch to evaluation mode and set device
    device = torch.device(deviceStr)
    model.eval()
    with torch.no_grad():
        #get for example 3 outputs
        validationCounter = 0
        for batch in validationDataSet:
            if(validationCounter == 3):
                break
            validationCounter +=1
            #get the encoder data
            encoderInput , inputMask = batch['encoderInput'], batch['inputMask']
            encoderInput , inputMask = encoderInput.to(device) , inputMask.to(device)
            inputText.append(batch['source'][0])
            targetText.append(batch['target'][0])
            #get model output
            output = generateTokenByToken(model=model,
                              inputData=encoderInput,inputMask=inputMask,
                              maximumSequenceLength=maxSequnceLength,
                              targetTokenizer=targetTokenizer)
            #output needs to be on cpu
            output = targetTokenizer.decode(np.array(output.detach().cpu()))
            predictedText.append(output)

    return inputText,targetText,predictedText


def train(pathToDataSet:str,splitTrain:float,sequenceLength:int,
          sourceLanguge:str='english',targetLanguge:str='croatian',
          modelEmbeddingDimension:float = 512,numberOfAttentionHeads:int =8,
          numberOfEncoderBlocks:int = 8,numberOfDecoderBlocks:int = 8,
          dimensionExpansionFactorInFeedForward:int=4,
          dropout:float=0.1,
          optimizer=torch.optim.Adam,learningRate = 0.001,
          lossFunction=torch.nn.CrossEntropyLoss,
          numberOfEpochs :int = 20,
          batchSize:int =10,
          continueTrain:bool = False,
          deviceStr:str = 'cuda',
          numberOfWorkers:int=8,
          learningRateDecay = False,
          aditionalTokenizedTrainingEveryOtherEpoch = False):
    #set device
    device = torch.device(deviceStr)

    #make the tokenizers
    inputWholeDataSet = pd.read_parquet(pathToDataSet)
    if(os.path.isdir('modelData') == False):
        tokenGenerate = \
            DatasetEx(inputRawDataSet=inputWholeDataSet,sequnceLegth=sequenceLength,
                      tokenizerCreated=False)
        tokenGenerate.tokenize()
        print('Tokenizers created, restart the script!')
        sys.exit(0)

    #load the tokenizers
    sourceLang = None
    targetLang = None
    for key in LANGUAGES.keys():
        if(LANGUAGES[key] == sourceLanguge):
            sourceLang = key
            break
    for key in LANGUAGES.keys():
        if(LANGUAGES[key] == targetLanguge):
            targetLang = key
            break
    sourceTokenizer = tokenizers.Tokenizer.from_file(f'modelData/tokenizer_{sourceLang}.json')
    targetTokenizer = tokenizers.Tokenizer.from_file(f'modelData/tokenizer_{targetLang}.json')
    #now we split data into train and validation
    trainDataSetSize = int(splitTrain*len(inputWholeDataSet['translation']))
    splitIndices = torch.randperm(len(inputWholeDataSet['translation']))
    trainDataRaw = inputWholeDataSet.iloc[splitIndices[:trainDataSetSize]]
    validationDataRaw = inputWholeDataSet.iloc[splitIndices[trainDataSetSize:]]
    trainData = \
        DatasetEx(inputRawDataSet=trainDataRaw,sequnceLegth=sequenceLength,tokenizerCreated=True,
                  tokenizerSource=sourceTokenizer, tokenizerTarget=targetTokenizer)
    validationData = \
        DatasetEx(inputRawDataSet=validationDataRaw,sequnceLegth=sequenceLength,tokenizerCreated=True,
                  tokenizerSource=sourceTokenizer, tokenizerTarget=targetTokenizer)
    
    #now make the dataLoader
    trainDataLoader = torchUtils.DataLoader(dataset=trainData,batch_size=batchSize, shuffle = True,
                                            num_workers=numberOfWorkers, pin_memory=True)
    validationDataLoader = torchUtils.DataLoader(dataset=validationData,
                                                 batch_size=1,
                                                 num_workers=numberOfWorkers, pin_memory=True)
    
    #we create a model now
    #first get vocabulary sizes
    inputVocabSize = sourceTokenizer.get_vocab_size()
    targetVocabSize = targetTokenizer.get_vocab_size()
    #now create a model
    model = TransformerModel(inputVocabularySize = inputVocabSize,inputSequenceLength=sequenceLength,
                             targetVocabularySize=targetVocabSize,targetSequenceLength=sequenceLength,
                             modelEmbeddingDimension=modelEmbeddingDimension,
                             numberOfAttentionHeads=numberOfAttentionHeads,
                             numberOfEncoderBlocks=numberOfEncoderBlocks,
                             numberOfDecoderBlocks=numberOfDecoderBlocks,
                             dropout=dropout,
                             dimensionExpansionFactorInFeedForward=dimensionExpansionFactorInFeedForward)
    model = model.to(device)
    #optimizer
    SelectedLearningRate = learningRate
    selectedOptimizer = optimizer(model.parameters(),lr=SelectedLearningRate,eps=1e-9) 
    #loss function, ignore [PAD] token
    selectedLossFunction = lossFunction(ignore_index=targetTokenizer.token_to_id('[PAD]'),
                                        label_smoothing=0.05)
    selectedLossFunction = selectedLossFunction.to(device)

    #if we want to continue train on pretrained
    if(continueTrain == True):
        print('Loading existing model and Continuing training.')
        modelState = torch.load('modelData/transformerModel')
        model.load_state_dict(modelState['modelStateDict'])
        selectedOptimizer.load_state_dict(modelState['optimizerStateDict'])
    else:
        #start new train
        print('Starting to train new model')
    #main train loop
    for epoch in range(numberOfEpochs):
        print(f'EPOCH: {epoch}')
        if(learningRateDecay == True):
            selectedOptimizer = \
                optimizer(model.parameters(),lr=SelectedLearningRate,eps=1e-9) 
            SelectedLearningRate = SelectedLearningRate/10
        #switch to train
        model.train()
        #iterate over train data batches
        for trainBatch in trainDataLoader:
            #get data
            encoderInput = trainBatch['encoderInput'].to(device)
            inputMask = trainBatch['inputMask'].to(device)
            decoderInput = trainBatch['decoderInput'].to(device)
            targetMask = trainBatch['targetMask'].to(device)
            label = trainBatch['label'].to(device)
            #zero out gradients
            selectedOptimizer.zero_grad(set_to_none=True)
            #now we train normaly or by mixing type
            if(aditionalTokenizedTrainingEveryOtherEpoch == True):
                #mixing
                #get the output from model normaly everry even epoch
                outputNormal = model.forward(inputToEncoder=encoderInput,
                                        inputToDecoder=decoderInput,
                                        targetMask=targetMask,
                                        inputMask=inputMask)
                #get token by token generated output
                outputTokenByToken = \
                    trainGeneratingTokenByToken(model=model,
                                                inputToEncoderData=encoderInput,
                                                inputMask=inputMask,
                                                targetTokenizer=targetTokenizer,
                                                targetVocabSize=targetVocabSize)
                #add them
                output = outputNormal + outputTokenByToken*10
            else:
            #just get the normal output from model
                output = model.forward(inputToEncoder=encoderInput,
                                       inputToDecoder=decoderInput,
                                       targetMask=targetMask,
                                       inputMask=inputMask)

            #calculate loss, reshape the outpus to 2d arrays
            outputReshaped = output.view(-1,targetVocabSize)
            labelReshaped = label.view(-1)
            loss = selectedLossFunction(outputReshaped,labelReshaped)
            print('\rLOSS:',loss.item(),end="")
            #backpropagation
            loss.backward()
            #optimizer step
            selectedOptimizer.step()

        #validation 
        #returns inputText,targetText,predictedText
        inputText,targetText,predictedText = \
            modelValidationTokenByToken(model=model,
                                        validationDataSet=validationDataLoader,
                                        maxSequnceLength=sequenceLength,
                                        targetTokenizer=targetTokenizer)
        print()
        print()
        print('validation results:')
        print()
        print('Source: ')
        print(inputText[0])
        print()
        print('Target: ')
        print(targetText[0])
        print()
        print('Prediction: ')
        print(predictedText[0])
        print()

    #save the model
    torch.save({
        'modelStateDict':model.state_dict(),
        'optimizerStateDict':selectedOptimizer.state_dict()
    },'modelData/transformerModel')


