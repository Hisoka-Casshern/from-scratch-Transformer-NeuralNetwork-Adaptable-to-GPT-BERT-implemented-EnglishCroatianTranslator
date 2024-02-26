'''this script implements classes for input embedding and positional encoding based
on the original transformer paper'''
import torch
import math

class Embedding(torch.nn.Module):
    def __init__(self, embeddingSize:int, vocabularySize:int):
        '''constructor creates a simple embedding using torch'''
        super().__init__()
        self.embeddingSize = embeddingSize
        self.vocabularySize = vocabularySize
        self.simpleEmbedding = torch.nn.Embedding(num_embeddings=vocabularySize,
                                                  embedding_dim=embeddingSize)
        
    def forward(self, inputTensor:torch.tensor):
        '''here we embed the input and also scale with second root of embed size,
        input has the shape(batch, lengthOfSequnce) and we make a new tensor of
        shape(batch, lengthOfSequnce, embeddingSize)'''
        embededInput = self.simpleEmbedding(inputTensor)
        embededInput = embededInput * math.sqrt(self.embeddingSize)
        return embededInput

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embeddingSize:int, lengthOfSequnce:int, dropout:float):
        super().__init__()
        '''constructor needs the embed size and length of sequnce to create
        a simple positional encoding using sine and cosine that will be added
        to input embedding, also initizes droput'''
        self.Dropout = torch.nn.Dropout(dropout)
        #first we create an array of integers correspoding to positions in a sequence,
        #we also give it a unit second dimension so we can broadcast correctly
        positionInASequnce = torch.arange(0,lengthOfSequnce, dtype=torch.float)
        positionInASequnce = positionInASequnce[:,None]
        #we create now positional buffer for whole embedding
        positionalEncoding = torch.zeros(lengthOfSequnce,embeddingSize)
        #now we use sine and cosine encoding
        for idx in range(positionalEncoding.shape[1]):
            #even columns are sine and odd cosine
            if(idx%2 == 0):
                positionalEncoding[:,idx] = \
                    (torch.sin(positionInASequnce/(10000**((2 * idx)/embeddingSize)))).flatten()
            else:
                positionalEncoding[:,idx] = \
                    (torch.cos(positionInASequnce/(10000**((2 * idx)/embeddingSize)))).flatten()
        #we add the batch dimension so we can broadcast over batch
        positionalEncoding = positionalEncoding[None,:,:]
        #register it with torch
        self.register_buffer('positionalEncoding',positionalEncoding, persistent=False)
    
    def forward(self, inputOntoWhichToAddPositionalEncoding: torch.Tensor):
        '''here we just add positional encoding to the input tensor,
        we account for different sequnce_length'''
        positionallyEncodedInput = inputOntoWhichToAddPositionalEncoding + \
            self.positionalEncoding[:,:inputOntoWhichToAddPositionalEncoding.shape[1],:]
        #pass through dropout
        positionallyEncodedInput = self.Dropout(positionallyEncodedInput)
        return positionallyEncodedInput





                
