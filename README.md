# E2E LID

End-to-end language identification from speech

## Requirements

```
Python = 3.6
Pytorch >= 1.0.0
Scikit-learn >=0.19
tqdm
h5py
```

## Prepare data

Data preparation scripts are provided and features in [Kaldi](https://kaldi-asr.org/) format are exepected.

Pre-processed data will consist of hdf files such that features for each recording are stored as datasets of shape [1, nfeat, nframes] and further stored under a group labeled with the language ID.

Prepare Kaldi features with data_prep.py. Arguments:

```
--path-to-data        Path to scp files with features
--data-info-path      Path to spk2utt and utt2spk
--spk2utt             Path to spk2utt
--utt2spk             Path to utt2spk
--path-to-more-data   Path to extra scp files with features
--more-data-info-path Path to spk2utt and utt2spk
--more-spk2utt Path   Path to spk2utt
--more-utt2spk Path   Path to utt2spk
--out-path Path       Path to output hdf file
--out-name Path       Output hdf file name
--min-recordings      Minimum number of train recordings for language to be included
```

Train and development data hdfs are expected.

## Train a model

Train models with train_olr.py. Arguments:

```
--batch-size          input batch size for training (default: 64)
--epochs              number of epochs to train (default: 500)
--lr                  learning rate (default: 0.001)
--alpha               RMSprop alpha (default: 0.99)
--l2                  Weight decay coefficient (default: 0.00001)
--swap                Swaps anchor and positive in case loss is higher that way
--checkpoint-epoch    epoch to load for checkpointing. If None, training starts from scratch
--checkpoint-path     Path for checkpointing
--pretrained-path     Path for pre trained model
--train-hdf-file      Path to hdf data
--valid-hdf-file      Path to hdf data
--model               {mfcc,fb,resnet_fb,resnet_mfcc,resnet_lstm,resnet_stats,lcnn9_mfcc,lcnn29_mfcc}
--workers             number of data loading workers
--seed                random seed (default: 1)
--save-every          how many epochs to wait before logging training status. Default is 1
--ncoef               number of MFCCs (default: 13)
--latent-size         latent layer dimension (default: 200)
--n-frames            maximum number of frames per utterance (default: 1000)
--n-cycles            cycles over speakers list to complete 1 epoch
--valid-n-cycles      cycles over speakers list to complete 1 epoch
--patience            Epochs to wait before decreasing LR (default: 30)
--softmax             {none,softmax,am_softmax}
--mine-triplets       Enables distance mining for triplets
--no-cuda             Disables GPU use
```

## Scoring test recordings

End-to-end scoring can be performed with eval_olr_sm.py, which will print EER in the screen and save scores for each trial in an output file.

Arguments:

```
--data-path           Path to input data
--sil-data            Path to input data with silence
 --trials-path        Path to trials file
--cp-path             Path for file containing model
--model               {mfcc,fb,resnet_fb,resnet_mfcc,resnet_lstm,resnet_stats,lcnn9_mfcc,lcnn29_mfcc}
--latent-size S       latent layer dimension (default: 200)
--ncoef N             number of MFCCs (default: 13)
--scores-file         Path for saving computed scores
--no-cuda             Disables GPU use
```

Scoring with other back-ends can be performed on top of embeddings which can be computed with embedd.py

## Citing

```
@article{monteiro2019residual,
  title={Residual Convolutional Neural Network with Attentive Feature Pooling for End-To-End Language Identification from Short-Duration Speech},
  author={Monteiro, Jo{\~a}o and Alam, Jahangir and Falk, Tiago H},
  journal={Computer Speech \& Language},
  year={2019},
  publisher={Elsevier}
}
```
