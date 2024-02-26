# from_scratch_Transformer_Neural_Network_Adaptable_to_GPT_or_BERT_English_to_Croatian_Translator
Using Pytorch I create from scratch a full Transformer model (with separate encoder, and decoder blocks that can then be stacked) as shown in the original [Transformer paper](https://arxiv.org/abs/1706.03762). This architecture can be easily then implemented to act as GPT or BERT. <br />
I here trained a full Transformer network to act as a translator from English to Croatian. Inside the EnglishCroatianDatabse is a database I created that has an English-to-Croatian pair of translated sentences, I created 1049773 such pairs and used them for my training but if needed anybody can use this database for their needs. These pairs are all low caps without punctuation <br />
There are several scripts in the script directory, the embedding script provides embedding and Positional Encoding classes, the transformerScript has the classes that define the encoder and decoder blocks and also the construction of the whole model using the embeddings class, train script does the training with a token by token generation for validation, the dataset creation script creates  from pandas data frame an object containing all needed inputs to the data loader such as input to the encoder, input to decoder, labels, and tasks tensors ready for batching also this class performs tokenization.<br />
The modelData directory contains .json tokenization for the English and Croatian sentences respectively and also contains a pre-trained model. <br />



<br />
Here I will shortly through images that show the workings of the transformer network and explain the process of the Transformer, images are all made by myself and are under a license that allows usage only for noncommercial and educational purposes:<br />

The Transformer Architecture is color-coded by layers:
<br />
<p align="center">
  <img src="GameScreenshots/0.png" width="300" height="400">
</p>


