'''Main script that contains the all functions and classes needed
to make a transformer using pytorch'''

import torch
from embeddingsScript import Embedding,PositionalEncoding
import math

#first we make a Attention class that we will use in encoder decoder blocks
class Attention(torch.nn.Module):
    def __init__(self, embeddingSize : int, numberOfHeads:int, dropout:float=None) -> None:
        '''constructor takes the model Embedding size that will be used in construction
        of linear layers that take query, keys and values'''
        
        #initialize the parent class constructor
        super().__init__()

        #store the local variables versions
        self.embedSize = embeddingSize
        self.numHeads = numberOfHeads

        #create droput
        self.dropout = torch.nn.Dropout(dropout)

        #check that we have proper number of heads and Embedding size
        if(embeddingSize%numberOfHeads != 0):
            #if not raise exception
            raise Exception("Embedding size is not divisible by the given number of heads")
        else:
            #store the size of each head
            self.headSize = int(embeddingSize/numberOfHeads)

        #now create a linear layers that will correspond to query, keys and values
        #these layers take input of shape(batchSize, lengthOfSequence, embeddingSize)
        #and return the transformed input tensors of shape(batchSize, lengthOfSequence, embeddingSize)
        #we also pass the final output of attention into output linear layer
        #these layers do not have bias
        self.QueriesLinearLayer = torch.nn.Linear(in_features=embeddingSize, out_features=embeddingSize, bias=False)
        self.KeysLinearLayer = torch.nn.Linear(in_features=embeddingSize, out_features=embeddingSize, bias=False)
        self.ValuesLinearLayer = torch.nn.Linear(in_features=embeddingSize, out_features=embeddingSize, bias=False)
        self.AttentionOutputLinear = torch.nn.Linear(in_features=embeddingSize, out_features=embeddingSize, bias=False)

    def forward(self, inputQueries:torch.tensor,inputKeys:torch.tensor,inputValues:torch.tensor,
                inputMask:torch.tensor = None):
        '''here we pass the inputs to linear layers, and then calculate the attention based on the 
        original transformer paper'''

        #get the q,k,v tensors
        Query = self.QueriesLinearLayer(inputQueries)
        Keys = self.KeysLinearLayer(inputKeys)
        Values = self.ValuesLinearLayer(inputValues)
        
        #now we split the tensors such that each head gets a portion of Embedding based on headSize
        #this gives shape(batchSize, lengthOfSequence, numberOfHeads, headSize)
        Query = Query.view(Query.shape[0], Query.shape[1], self.numHeads, self.headSize)
        Keys = Keys.view(Keys.shape[0], Keys.shape[1], self.numHeads, self.headSize)
        Values = Values.view(Values.shape[0], Values.shape[1], self.numHeads, self.headSize)

        #to perform attention we need to perform dot product with respect to sequence length and the 
        #the embed of each head (head size) for each head and batch independantly so we transpose (permute)
        #this gives shape(batchSize, numberOfHeads, lengthOfSequence, headSize) 
        Query = Query.permute(0,2,1,3)
        Keys = Keys.permute(0,2,1,3)
        Values = Values.permute(0,2,1,3)

        #now we perform attention 
        #firstly we matrix mutliplicate Query and Keys (transpose last two keys dimension) 
        #if mask givem we apply the mask before softmax
        #the shape we get is (batchSize, numberOfHeads, lengthOfSequence, lengthOfSequence) 
        #so the softmax is applied for each column
        #finally we matrix multiply it with values which gives 
        #shape(batchSize, numberOfHeads,lengthOfSequence, headSize) 
        calculatedAttention = torch.matmul(Query,Keys.permute(0,1,3,2))/math.sqrt(self.headSize)
        if(inputMask != None):
            calculatedAttention.masked_fill(inputMask == 0, float('-inf'))
        calculatedAttention = torch.nn.functional.softmax(calculatedAttention, dim=-1)
        if(self.dropout != None):
            calculatedAttention = self.dropout(calculatedAttention)    
        calculatedAttention = torch.matmul(calculatedAttention,Values)

        #now we need to concatenate back into embed size of model by permuting back sequnce lenght with
        #number of heads and reducing the last two dimensions into one
        #this gives shape(batchSize,lengthOfSequence, embeddingSize)
        calculatedAttention = calculatedAttention.permute(0,2,1,3)
        calculatedAttention = calculatedAttention.contiguous().view(calculatedAttention.shape[0],-1,
                                                                    self.numHeads * self.headSize)
        #pass this input into output layer and return the output of same shape
        attentionOutput = self.AttentionOutputLinear(calculatedAttention)

        return attentionOutput
    

#Define encoder and decoder blocks, we separate them into different classes for modularity
#this will allow us to later be able to stack these blocks in build model
class EncoderUnitBlock(torch.nn.Module):
    def __init__(self,modelEmbeddingSize:int,selfAttentionMechanism:Attention, 
                 layerExpansion :int, dropout:float) -> None:
        '''constructor takes attention class (in this case acts as selfattention), and constructs 
        other linear layers as defined in original transformer paper. Layer expansion is used to 
        expand the first linear layer (expanded layer) in feed forward part'''
        super(EncoderUnitBlock,self).__init__()

        #store local self attention
        self.ThisBlocksSelfAttentionMechanism = selfAttentionMechanism
        #create the feed forward layers and output
        self.ExpandedLinearLayer = torch.nn.Linear(modelEmbeddingSize,int(modelEmbeddingSize*layerExpansion))
        self.DropoutExpandedLinearLayer= torch.nn.Dropout(dropout)
        self.OutputCompressionLayer = torch.nn.Linear(int(modelEmbeddingSize*layerExpansion),modelEmbeddingSize)
        #normalization and dropout for each part output for implementation of residual connections
        self.FirstNormalization = torch.nn.LayerNorm(modelEmbeddingSize)
        self.SecondNormalization = torch.nn.LayerNorm(modelEmbeddingSize)
        self.FirstOutputLayerDroput = torch.nn.Dropout(dropout)
        self.SecondOutputLayerDroput = torch.nn.Dropout(dropout)

    def forward(self,encoderInputTensor:torch.tensor,encoderMask:torch.tensor):
        #firstly get the output from attention
        #normalize output layer, in new way of doing this we normalize inputs
        # then calculate self Attention and add droput
        normalizedInputToAttention = self.FirstNormalization(encoderInputTensor)
        attentionMechanismOutput = \
            self.ThisBlocksSelfAttentionMechanism.forward(inputQueries=normalizedInputToAttention,
                                                          inputKeys=normalizedInputToAttention,
                                                          inputValues=normalizedInputToAttention,
                                                          inputMask=encoderMask)
        attentionMechanismOutput = self.FirstOutputLayerDroput(attentionMechanismOutput)

        #first residual connection
        #add the output to the input connection
        attentionOutputAfterNormAndResid = encoderInputTensor + attentionMechanismOutput

        #now we perform feed forward part
        #layer by layer, so we first pass to expanded layer data and as in paper use relu then droput, 
        #data is here normalized instead at output
        #then to the compression layer meaning the output layer
        normalizedInputToFeedForward = self.SecondNormalization(attentionOutputAfterNormAndResid)
        feedForwardOutput = self.ExpandedLinearLayer(normalizedInputToFeedForward)
        feedForwardOutput = torch.nn.functional.relu(feedForwardOutput)
        feedForwardOutput = self.DropoutExpandedLinearLayer(feedForwardOutput)
        feedForwardOutput = self.OutputCompressionLayer(feedForwardOutput)
        feedForwardOutput = self.SecondOutputLayerDroput(feedForwardOutput)

        #finally add the second residual connection and return
        output = attentionOutputAfterNormAndResid + feedForwardOutput       

        return output


class DecoderUnitBlock(torch.nn.Module):
    def __init__(self,modelEmbeddingSize:int,
                selfAttentionMechanism:Attention,crossAttentionMechanism:Attention,
                layerExpansion :int, dropout:float) -> None:
        '''constructor takes  two attention classes where first is used ad self attention and second
        as cross attention with encoder block outputs, and constructs other linear layers as defined in 
        original transformer paper. Layer expansion is used to expand the first linear layer 
        (expanded layer) in feed forward part'''
        super(DecoderUnitBlock,self).__init__()
      
        #store local self and cross  attention
        self.ThisBlocksSelfAttentionMechanism = selfAttentionMechanism
        self.ThisBlocksCrossAttentionMechanism = crossAttentionMechanism
        #create the feed forward layers and output
        self.ExpandedLinearLayer = torch.nn.Linear(modelEmbeddingSize,int(modelEmbeddingSize*layerExpansion))
        self.DropoutExpandedLinearLayer= torch.nn.Dropout(dropout)
        self.OutputCompressionLayer = torch.nn.Linear(int(modelEmbeddingSize*layerExpansion),modelEmbeddingSize)
        #normalization and dropout for each part output for implementation of residual connections
        self.FirstNormalization = torch.nn.LayerNorm(modelEmbeddingSize)
        self.SecondNormalization = torch.nn.LayerNorm(modelEmbeddingSize)
        self.ThirdNormalization = torch.nn.LayerNorm(modelEmbeddingSize)
        self.FirstOutputLayerDroput = torch.nn.Dropout(dropout)
        self.SecondOutputLayerDroput = torch.nn.Dropout(dropout)
        self.ThirdOutputLayerDroput = torch.nn.Dropout(dropout)

    def forward(self,decoderInputTensor:torch.tensor,encoderOutputTensor:torch.tensor,
                decoderMask:torch.tensor, encoderMask:torch.tensor):
        #firstly get the output from decoder self attention
        #normalize output layer, in new way of doing this we normalize inputs
        #then calculate self Attention and add droput
        normalizedInputToSelfAttention = self.FirstNormalization(decoderInputTensor)
        selfAttentionMechanismOutput = \
            self.ThisBlocksSelfAttentionMechanism.forward(inputQueries=normalizedInputToSelfAttention,
                                                          inputKeys=normalizedInputToSelfAttention,
                                                          inputValues=normalizedInputToSelfAttention,
                                                          inputMask=decoderMask)
        selfAttentionMechanismOutput = self.FirstOutputLayerDroput(selfAttentionMechanismOutput)

        #first residual connection
        #add the output to the input connection
        selfAttentionOutputAfterNormAndResid = \
            decoderInputTensor + selfAttentionMechanismOutput

        #now we do the same but pefrorm cross attention with encoder
        normalizedInputToCrossAttention = self.SecondNormalization(selfAttentionOutputAfterNormAndResid)
        crossAttentionMechanismOutput = \
            self.ThisBlocksCrossAttentionMechanism.forward(inputQueries=normalizedInputToCrossAttention,
                                                           inputKeys=encoderOutputTensor,
                                                           inputValues=encoderOutputTensor,
                                                           inputMask=encoderMask)
        crossAttentionMechanismOutput = self.SecondOutputLayerDroput(crossAttentionMechanismOutput)

        #second residual connection
        #add the output to the input connection
        crossAttentionOutputAfterNormAndResid = \
            selfAttentionOutputAfterNormAndResid + crossAttentionMechanismOutput

        #now we perform feed forward part
        #layer by layer, so we first pass to expanded layer data and as in paper use relu then droput, 
        #data is here normalized instead at output
        #then to the compression layer meaning the output layer
        normalizedInputToFeedForward = self.ThirdNormalization(crossAttentionOutputAfterNormAndResid)
        feedForwardOutput = self.ExpandedLinearLayer(normalizedInputToFeedForward)
        feedForwardOutput = torch.nn.functional.relu(feedForwardOutput)
        feedForwardOutput = self.DropoutExpandedLinearLayer(feedForwardOutput)
        feedForwardOutput = self.OutputCompressionLayer(feedForwardOutput)
        feedForwardOutput = self.SecondOutputLayerDroput(feedForwardOutput)

        #finally add the second residual connection and return
        output = crossAttentionOutputAfterNormAndResid + feedForwardOutput
        
        return output
    
        
#we create the main model class that will be built using all the parts and returned
#this class can then be used to call the functions that calculate encoder, decoder and final projection
#outputs, also implements the 
class TransformerModel(torch.nn.Module):
    def __init__(self, inputVocabularySize:int, inputSequenceLength:int,
                targetVocabularySize:int, targetSequenceLength:int, modelEmbeddingDimension:float,
                numberOfAttentionHeads:int, numberOfEncoderBlocks:int, numberOfDecoderBlocks:int,
                dropout: float, dimensionExpansionFactorInFeedForward:int):
        super().__init__()
        '''constructor initializes embedd an positional class and stores local variables'''
        #local store
        self.inputVocabSize = inputVocabularySize
        self.inputSeqLen = inputSequenceLength
        self.targetVocabSize = targetVocabularySize
        self.targetSeqLen = targetSequenceLength
        self.modelEmbeddDim = modelEmbeddingDimension
        self.numHeads = numberOfAttentionHeads
        self.dropout = dropout
        self.expandFactor =dimensionExpansionFactorInFeedForward

        #we initialize embedding and positional encoding classes
        self.inputEmbedding = Embedding(embeddingSize=self.modelEmbeddDim, 
                                        vocabularySize=self.inputVocabSize)
        self.inputPositionalEncoding = PositionalEncoding(embeddingSize=self.modelEmbeddDim,
                                                     lengthOfSequnce=self.inputSeqLen,
                                                     dropout=self.dropout)
        self.targetEmbedding = Embedding(embeddingSize=self.modelEmbeddDim,
                                        vocabularySize=self.targetVocabSize)
        self.targetPositionalEncoding = PositionalEncoding(embeddingSize=self.modelEmbeddDim,
                                                     lengthOfSequnce=self.targetSeqLen,
                                                     dropout=self.dropout)
       
        #now we create a list of encoders
        self.encoderBlocksList = []
        for _ in range(numberOfEncoderBlocks):
            #create for each block its own attention mechanism
            selfAttention = Attention(embeddingSize=self.modelEmbeddDim,numberOfHeads=self.numHeads,
                                  dropout=self.dropout)
            encoderBlock = EncoderUnitBlock(modelEmbeddingSize=self.modelEmbeddDim,selfAttentionMechanism=selfAttention,
                                            layerExpansion=self.expandFactor,dropout=self.dropout)
            self.encoderBlocksList.append(encoderBlock)
        self.encoderBlocksList = torch.nn.ModuleList(self.encoderBlocksList)
        # also we create a norm layer for last out layer 
        self.encoderLayerNorm = torch.nn.LayerNorm(self.modelEmbeddDim)

        #now we create a list of decoders
        self.decoderBlocksList = []
        for _ in range(numberOfDecoderBlocks):
            #create for each block its own attention mechanism
            selfAttention = Attention(embeddingSize=self.modelEmbeddDim,numberOfHeads=self.numHeads,
                                  dropout=self.dropout)
            crossAttention = Attention(embeddingSize=self.modelEmbeddDim,numberOfHeads=self.numHeads,
                                  dropout=self.dropout)
            decoderBlock = DecoderUnitBlock(modelEmbeddingSize=self.modelEmbeddDim,
                                            selfAttentionMechanism=selfAttention,
                                            crossAttentionMechanism=crossAttention,
                                            layerExpansion=self.expandFactor,dropout=self.dropout)
            self.decoderBlocksList.append(decoderBlock)
        self.decoderBlocksList = torch.nn.ModuleList(self.decoderBlocksList)
        # also we create a norm layer for last out layer 
        self.decoderLayerNorm = torch.nn.LayerNorm(self.modelEmbeddDim)

        #we also create a linear layer for the last blcok that represent the projection which
        #takes data from decoder and projects to target vocabulary space
        self.projectionLayer = \
            torch.nn.Linear(in_features=self.modelEmbeddDim, out_features=self.targetVocabSize)

    #now we define the function that will run data through full encode layer
    def transformerEncoder(self, inputToEncoder:torch.tensor, inputMask:torch.tensor):
        #encode inputs
        inputData = self.inputEmbedding.forward(inputTensor=inputToEncoder)
        inputData = \
            self.inputPositionalEncoding.forward(inputOntoWhichToAddPositionalEncoding=inputData)
        #now we push input data from first to last encoder block
        for block in self.encoderBlocksList:
            inputData = block.forward(encoderInputTensor=inputData,encoderMask=inputMask)
        #we layer normalize last output 
        outData = self.encoderLayerNorm(inputData)
        return outData

    #now we define the function that will run data through full decoder layer
    def transformerDecoder(self, inputToDecoder:torch.tensor, targetMask:torch.tensor,
                           outputFromEncoder:torch.tensor, inputMask:torch.tensor):
        #encode targets
        inputData = self.targetEmbedding.forward(inputTensor=inputToDecoder)
        inputData = \
            self.targetPositionalEncoding.forward(inputOntoWhichToAddPositionalEncoding=inputData)
        #now we push input data from first to last decoder block
        for block in self.decoderBlocksList:
            inputData = block.forward(decoderInputTensor=inputData,
                                      encoderOutputTensor=outputFromEncoder,
                                      decoderMask=targetMask,encoderMask=inputMask)
        #we layer normalize last output 
        outData = self.decoderLayerNorm(inputData)
        return outData

    #now we define the last block which is a projection from the embedd 
    #dimension to target vocabulary dimension
    def transformerProjection(self, outputFromDecoder:torch.tensor):
        projectionOutput = self.projectionLayer(outputFromDecoder)
        return projectionOutput
    
    #finally we have a forward method that runs whole transformer
    def forward(self,inputToEncoder:torch.tensor,inputToDecoder:torch.tensor, 
                targetMask:torch.tensor, inputMask:torch.tensor):
        #first we pass through encoder
        encoderOutput = self.transformerEncoder(inputToEncoder=inputToEncoder,
                                                inputMask=inputMask)

        #now through decoder
        decoderOutput = self.transformerDecoder(inputToDecoder=inputToDecoder,
                                                targetMask=targetMask,
                                                outputFromEncoder=encoderOutput,
                                                inputMask=inputMask)
        #finally the projection
        output = self.transformerProjection(outputFromDecoder=decoderOutput)

        return output