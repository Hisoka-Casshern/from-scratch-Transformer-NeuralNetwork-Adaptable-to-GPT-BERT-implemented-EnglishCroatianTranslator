'''here we define a class that will create dataset for a translation between two
languages, it will incorporate get and len methods so we are able to use
pytrch dataloder easily, we use the tokenzer as descibed in huggingface.co'''

import torch
from torch.utils.data import Dataset
import tokenizers
from googletrans import LANGUAGES
import os
from pathlib import Path
import sys


class DatasetEx(Dataset):
    def __init__(self,inputRawDataSet, sequnceLegth:int,
                 sourceLang:str='english',targetLanguage:str='croatian',
                 tokenizerCreated:bool = False ,
                 tokenizerSource:tokenizers.Tokenizer=None,
                 tokenizerTarget:tokenizers.Tokenizer = None):
        super().__init__()
        '''constructor takes raw parquete data formated such as
        row=['index':index,'translation'{'sourceLang':...,'targetLang':}
        also needs a sequence length so we can define model max sequnce
        and pad or cut'''
        #set local directory path
        os.chdir(Path(__file__).parent)
        #local storage
        self.data = inputRawDataSet
        self.seqLen = sequnceLegth
        for key, value in LANGUAGES.items():
            if(value == sourceLang):
                self.sourceLang = key
        for key, value in LANGUAGES.items():
            if(value == targetLanguage):
                self.targetLanguage = key
        #if we have already a tokenizer created on the dataset we can then load it
        #tokenizer needs to be run on whole dataset
        if(tokenizerCreated == True):
            #load the tokenizers
            self.tokenizerSource = tokenizerSource
            self.tokenizerTarget = tokenizerTarget
            #create local special tokens
            #source
            self.SourceSOStoken = \
                torch.tensor([self.tokenizerSource.token_to_id('[SOS]')],dtype=torch.int64)
            self.SourcePADtoken = \
                torch.tensor([self.tokenizerSource.token_to_id('[PAD]')],dtype=torch.int64)
            self.SourceEOStoken = \
                torch.tensor([self.tokenizerSource.token_to_id('[EOS]')],dtype=torch.int64) 
            #target
            self.TargetSOStoken = \
                torch.tensor([self.tokenizerTarget.token_to_id('[SOS]')],dtype=torch.int64)
            self.TargetPADtoken = \
                torch.tensor([self.tokenizerTarget.token_to_id('[PAD]')],dtype=torch.int64)
            self.TargetEOStoken = \
                torch.tensor([self.tokenizerTarget.token_to_id('[EOS]')],dtype=torch.int64) 

    def tokenize(self):
        '''checks modelData directory if there exists already a tokenizer and if not
        saves a new new tokenzies the inputRawDataSet.
        Tokenizer is saved as .json file for each language'''
        if(os.path.isfile(f'modelData/tokenizer_{self.sourceLang}.json') == False
           or os.path.isfile(f'modelData/tokenizer_{self.targetLanguage}.json') == False):
            #definee models
            modelSource = tokenizers.models.WordLevel
            modelTarget = tokenizers.models.WordLevel
            #create tokenizers and put the [UNK] if not found
            self.tokenizerSource = tokenizers.Tokenizer(model=modelSource(unk_token='[UNK]'))
            self.tokenizerTarget = tokenizers.Tokenizer(model=modelTarget(unk_token='[UNK]'))
            #define preTokenizer with whitespace as delimiter
            self.tokenizerSource.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
            self.tokenizerTarget.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
            #define trainers by words level
            trainerSource = tokenizers.trainers.WordLevelTrainer(
                special_tokens = ['[UNK]','[PAD]','[SOS]','[EOS]'],min_frequency = 1)
            trainerTarget = tokenizers.trainers.WordLevelTrainer(
                special_tokens = ['[UNK]','[PAD]','[SOS]','[EOS]'],min_frequency = 1)
            #create iterators of sentences
            sourceSetencesIterator = \
                iter([sentence[self.sourceLang] for sentence in list(self.data['translation'])])
            targetSetencesIterator = \
                iter([sentence[self.targetLanguage] for sentence in list(self.data['translation'])])
            #train the tokenizers
            self.tokenizerSource.train_from_iterator(iterator=sourceSetencesIterator,trainer=trainerSource)
            self.tokenizerTarget.train_from_iterator(iterator=targetSetencesIterator,trainer=trainerTarget)
            #create a director if onot existant
            if(os.path.isdir('modelData') == False):
                os.mkdir('modelData')
            #save the tokenizers
            self.tokenizerSource.save(f'modelData/tokenizer_{self.sourceLang}.json')
            self.tokenizerTarget.save(f'modelData/tokenizer_{self.targetLanguage}.json')
        else:
            print('Tokenizer already exists in the modelData directory!')

    def __len__(self):
        #get the size of data in loaded into RAM
        return len(self.data)
    
    def __getitem__(self, index):
        #get source and target texts
        sentences= self.data.iloc[index]['translation']
        sourceText = sentences[self.sourceLang]
        targetText = sentences[self.targetLanguage]
        #tokenize them
        sourceTextTokenized = \
            torch.tensor(self.tokenizerSource.encode(sourceText).ids,dtype=torch.int64)
        targetTextTokenized = \
            torch.tensor(self.tokenizerTarget.encode(targetText).ids,dtype=torch.int64)
        #check for error in case on long sentence
        if((self.seqLen - len(sourceTextTokenized) < 0) or
           (self.seqLen - len(targetTextTokenized) < 0)):
            sys.exit("Sentence longer than defined Sequence Length")
        #now we need to pad the sequences with special tokens defined
        #[PAD] token is added if the sequence is shorter than defined length of sentence
        #for source to get the number of [PAD] tokens we also need to subtract one [SOS] and one [EOS]
        #for target we only account for [EOS]
        numOfPADtokensForSource = self.seqLen - len(sourceTextTokenized) - 1 - 1
        numOfPADtokensForTarget = self.seqLen - len(targetTextTokenized) - 1
        PADtokensForSource = \
            torch.tensor([self.SourcePADtoken]*numOfPADtokensForSource,dtype=torch.int64)
        PADtokensForTarget = \
            torch.tensor([self.TargetPADtoken]*numOfPADtokensForTarget)

        #now we make encoder , decoder inputs and labels
        #encoder inputs are source sentences, we put also [SOS] and [EOS]
        encoderInput = \
            torch.cat([self.SourceSOStoken,
                       sourceTextTokenized,
                       self.SourceEOStoken,
                       PADtokensForSource],dim=0)

        #decoder inputs are translated senteces, we put only [SOS]
        decoderInput = \
            torch.cat([self.TargetSOStoken,
                       targetTextTokenized,
                       PADtokensForTarget],dim=0)

        #labels are translated senteces, we put only [EOS]
        label = \
            torch.cat([targetTextTokenized,
                       self.TargetEOStoken,
                       PADtokensForTarget],dim=0)

        #create masks and make sure [PAD] tokens are not used for learning by nulifying them
        #input mask is whole tensor just without [PAD]
        inputMask = (encoderInput != self.SourcePADtoken).int()
        inputMask = inputMask[None,None,:]
        #target mask is lower triangular brodacasted bitwisedAnd with [PAD] nulified tensor
        targetMask = (decoderInput != self.TargetPADtoken).int()
        targetMask = targetMask[None,:]
        targetLowerTraingular = \
            torch.tril(torch.ones(1,decoderInput.size(0),decoderInput.size(0),dtype=torch.int))
        targetMask = targetMask & targetLowerTraingular
        
        #make return dictionary
        returnDict = {'encoderInput': encoderInput, 'inputMask':inputMask,
                      'decoderInput': decoderInput, 'targetMask':targetMask,
                      'label': label, 'source':sourceText, 'target':targetText}
        
        return returnDict
    
    
