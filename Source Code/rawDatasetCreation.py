import os
from pathlib import Path
import datasets
import pandas as pd
from multiprocessing import Process, Queue
import time
from deep_translator import GoogleTranslator

class DesiredLangugeRawDataset:
    def __init__(self,listOfHuggingFaceDatasets:dict[str,str] =
                 {'opus_books':'en-hu',
                  'generics_kb':'generics_kb_best'}):
        '''constructor takes a dictionary of datasets which will be downloaded
        , for this purpse we care only about those that contain english 
        senteces and thus will use the default dictionary, for each other
        entry getter needs to be adjusted for the formating'''
        self.dictOfDatasets = listOfHuggingFaceDatasets
        os.chdir(Path(__file__).parent)

    @staticmethod
    def translate(stringToTranslate:str, 
                  sourceLanguage='english', 
                  targetLanguage='croatian'):
        '''we define a function that takes a string and using google
        engine translates it, this is needed later when generating a desired 
        dataset, you can just put languges of your desire'''
        translation = GoogleTranslator(source=sourceLanguage, target=targetLanguage)
        translation = translation.translate(stringToTranslate)
        return translation
 
    @staticmethod
    def processChunk(queue:Queue, dataChunk, processId):
        #multiprocessing kernel
        print('\n',f'Process {processId} Started Translating:')
        sourceAndTargetTranslationListOfDicts = []
        translationProgressCounter = 0
        for sentence in dataChunk:
            #track progress of one process as others have the same length
            if(processId == 0):
                print(f'\r progress of one process: {translationProgressCounter}/{len(dataChunk)}',end="")
            translationProgressCounter+=1
            #translate
            translation = DesiredLangugeRawDataset.translate(stringToTranslate=sentence,
                                                    sourceLanguage='english',
                                                    targetLanguage='croatian')
            #create a dictionary pair and append it to the main buffer
            translationDict = {'translation':{'en':sentence, 'hr':translation}}
            sourceAndTargetTranslationListOfDicts.append(translationDict)
        #add to the queue the solution
        print('\n',f'Process {processId} Ended Translating:')
        queue.put(sourceAndTargetTranslationListOfDicts)

    def generateDatasetFromListOfDatasets(self):
        '''save a copy of downloaded data inside RawDatasets a directory, and
        in a new RawDataset save a .parquet file that is organized as
        opus_books dataset just with new lagnuge as translation target '''
        if(os.path.isdir('RawDatasets') == False):
            os.mkdir('RawDatasets')
            #get the datasets and store them as .parquete
            for datasetPathName,datasetDataName in self.dictOfDatasets.items():
                dataset =datasets.load_dataset(path=datasetPathName,name=datasetDataName,split='train')
                dataset.to_parquet(f'RawDatasets/{datasetPathName}')
        else:
            print()
            print('Raw untranslated Data already present and will be used.')

        #now we create a list of dictionaries that will be used to store final raw pandas dataframe
        #these dictionaries will have source and target pairs translations
        sourceAndTargetTranslationListOfDicts = []

        #Next part If needed adjust for different cases
        #in our case we will first extract english sentecens from opus_books and then from generics_kb
        #then we translate each entence and store the dictionaries with original english sentence and 
        #opusBooks
        opusBooks = pd.read_parquet(f'RawDatasets/{list(self.dictOfDatasets.keys())[0]}')
        opusBooksEnglishSenteces = \
            [opusBooks['translation'][idx]['en'] for idx in range(len(opusBooks['translation']))]
        #genericsKb
        genericsKb = pd.read_parquet(f'RawDatasets/{list(self.dictOfDatasets.keys())[1]}')
        genericsKbEnglishSentences = list(genericsKb['generic_sentence'])
        #buffer of english senteces
        listOfAllEnglishSenteces = []
        listOfAllEnglishSenteces.extend(opusBooksEnglishSenteces)
        listOfAllEnglishSenteces.extend(genericsKbEnglishSentences)
    
        #translation loop
        numberOfSentecesToTranslate = len(listOfAllEnglishSenteces)
        #I use a multiprocesssing module to split the data over different processes
        #first get the number of processes avaliable
        numProcesses = os.cpu_count()
        sizeOfChunk = int(numberOfSentecesToTranslate/numProcesses)
        chunksOfDataForEachProcess = []
        for idx in range(0,numberOfSentecesToTranslate,sizeOfChunk):
            chunksOfDataForEachProcess.append(listOfAllEnglishSenteces[idx:idx+sizeOfChunk])
        #in case we have more chunks than processes add last chunk to last process (its just one data sample)
        if(len(chunksOfDataForEachProcess) > numProcesses):
            lastChunk = chunksOfDataForEachProcess.pop()
            chunksOfDataForEachProcess[-1].extend(lastChunk)
        #create a queue
        queue = Queue()
        #create process for each chunk
        processes = \
            {idx:Process(target=DesiredLangugeRawDataset.processChunk, args=(queue,chunk,idx)) 
                                    for idx,chunk in enumerate(chunksOfDataForEachProcess)}
        #start each process
        [process.start() for _,process in processes.items()]
        #buffer to collect list of dictionaires
        processesOutput = []
        while any(process.is_alive() for _,process in processes.items() or queue.empty() != False):
            while(queue.empty() != False):
                output = queue.get()

                processesOutput.append(output)
            time.sleep(0.2)
        
        #now we create a pandas dataframe that will be saved in a directory EnglishCroatianDatabase
        #you can here change and implement your own desired directory name
        sourceAndTargetTranslationListOfDicts = \
            [{'index':idx,'translation':translation} for idx,translation in enumerate(output)]
        dataframe = pd.DataFrame(sourceAndTargetTranslationListOfDicts)
        if(os.path.isdir('EnglishCroatianDatabase') == False):
            os.mkdir('EnglishCroatianDatabase')
        dataframe.to_parquet('EnglishCroatianDatabase/en_hr_translationsDatabase')
