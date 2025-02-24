# Light-Weight Vision Language Model Guided Gesture Recognition Based on Electromyography

### VLM for Gesture Classification

## Overview

<p align="center">
    <img src="images/1.png" style="max-width:500px">
</p>


>**Abstract**: <br>
> Active rehabilitation of the upper limbs requires the decoding of user intentions from biosensors, particularly electromyography (EMG), to deliver augmented support during activities of daily living (ADL). Traditional methods typically depend on supervised, subject-specific training to reduce accuracy degradation caused by anatomical and environmental variations. However, these approaches often neglect the context in which gestures are intended. To bridge this gap, we propose a novel gesture decoding pipeline that integrates vision understanding to provide contextual information, thereby enhancing the control of objects during gesture inference. Our decoding pipeline is optimized for edge deployment and is evaluated for its ability to infer gestures with high classification accuracy and reliability.

## Citation
If you make use of our work, please cite our paper:


```bibtex
This paper is currently under consideration. 
```


## Result Presented in the Paper

'''
1. The FAN model pre-trained on 33 participants can be found [here](https://github.com/deremustapha/VLM-EMG4Gesture/tree/master/code_result/1_FAN_Base_Decoder_33.ipynb).
2. The model with replacement of the FAN layer pre-trained on 33 participants can be found in [here](https://github.com/deremustapha/VLM-EMG4Gesture/tree/master/code_result/1_EMGNet_Base_Decoder_N.ipynb).
3. The fine-tuned model results can be found [here](https://github.com/deremustapha/VLM-EMG4Gesture/tree/master/code_result).


## Getting Started

We recommend using the [**Anaconda**](https://www.anaconda.com/) package manager to avoid dependency/reproducibility
problems.


## Data

1. The data used for pre-training the model can be downloaded from [online](https://ieee-dataport.org/documents/emg-eeg-dataset-upper-limb-gesture-classification).
2. The new participant dataset can be gotten [here](https://github.com/deremustapha/VLM-EMG4Gesture/tree/master/data)



## Setting Up the Paligemma 
This process can take sometime. Hence we recommend following the instructions from the source [page](https://huggingface.co/blog/paligemma). 


## Helper Files

1. Data Preparation
```sh
get the from src/data_preparation.py
```


2. Data Preprocessing 
```sh
get the from src/preprocessing.py
```

## Acknowledgements

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT), (No. RS-2023-00277220)
