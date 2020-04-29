# Model Overview

`3 Models` 

**1. LGBM**  
It is an upgraded model from the public kernels. 
More features, data and periods were fed to the model.

**2. CNN+DNN**  
This is a traditional NN model, where the CNN part is a dilated causal convolution inspired by WaveNet, and the DNN part is 2 FC layers connected to raw sales sequences. Then the inputs are concatenated together with categorical embeddings and future promotions, and directly output to 16 future days of predictions.

**3. RNN**  
This is a seq2seq model with a similar architecture of @Arthur Suilin's solution for the web traffic prediction. Encoder and decoder are both GRUs. The hidden states of the encoder are passed to the decoder through an FC layer connector. This is useful to improve the accuracy significantly.


# How to Run the Models

- cnn.py
- lgbm.py
- seq2seq.py

1. Download the data from the competition website.   
2. Add records of 0 with any existing store-item combo on 
every Dec 25th in the training data.   
3. Use the function *load_data()* in Utils.py to load and transform 
the raw data files.  
4. Use *save_unstack()* to save them to feather files. 
5. In the model codes, change the input of *load_unstack()* to 
the filename you saved. 
6. Then the models can be runned. 
