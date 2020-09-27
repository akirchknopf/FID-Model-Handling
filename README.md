

# FID-Model-Handling (Calculations and Experiments)

## Description

This is the repository for the master thesis 'Automated Identification of Information Disorder in Social Media from Multimodal Data'. With the help of [NetIdee](www.netidee.at) successfully implemented. 

This is the second repository of the master thesis. Its main purpose is the model experiment part. The other repositories are:

 - [Preprocessing of the Fakeddit Dataset](https://github.com/akirchknopf/FID-Preprocessing)
 - [Model Evaluation](https://github.com/akirchknopf/FID-Evaluation) 


## License


* This project is licensed under the GNU General Public License version 3 (GPL v3) - see the [GPL.txt](gpl.txt) file for details.
* This document is distributed under CC-BY-Sharelike-3.0 AT

## Installation Instructions

### Virtual Environment
```
python3 -m venv ./venv

source venv/bin/activate

pip install --upgrade pip

pip3 install jupyter

pip3 install tensorflow-gpu==2.3.0

pip3 install pandas

pip3 install bert-for-tf2

pip3 install scikit-learn

pip3 install telegram_send
# Configure telegram_send for retrieving status information about training according to: 
[Documentation about telegram send](https://pypi.org/project/telegram-send/) 


```
## Directory Structure
```
.
├── final_models.py
├── gpl.txt
├── models
│   └── best_models
│       ├── single_meta
│       │   └── weights-improvement-100-0.62.hdf5
│       ├── single_text_comments
│       │   └── weights-improvement-03-0.87.hdf5
│       ├── single_text_title
│       │   └── weights-improvement-02-0.88.hdf5
│       └── single_visual
│           └── weights-improvement-02-0.81.hdf5
├── multi_cased_L-12_H-768_A-12
│   ├── bert_config.json
│   ├── bert_model.ckpt.data-00000-of-00001
│   ├── bert_model.ckpt.index
│   ├── bert_model.ckpt.meta
│   └── vocab.txt
├── README.md
├── requirements.txt
├── text_sequence_analysis.ipynb
├── training_comments_bert_preset.ipynb
├── training_comments_image_preset.ipynb
├── training_image_inceptionv3_preset.ipynb
├── training_image_ResNet101V2_preset.ipynb
├── training_image_resnet_50v2_preset.ipynb
├── training_meta_image_preset.ipynb
├── training_meta_preset.ipynb
├── training_text_bert_preset.ipynb
├── training_text_comments_meta_preset.ipynb
├── training_text_comments_visual_meta_preset.ipynb
├── training_text_image_preset.ipynb
├── training_text_title_comments_meta_preset.ipynb
├── training_text_title_comments_preset.ipynb
├── training_text_title_comments_visual_meta_Add_preset.ipynb
├── training_text_title_comments_visual_meta_Maximum_preset.ipynb
├── training_text_title_comments_visual_meta_preset_bakk.ipynb
├── training_text_title_comments_visual_meta_preset.ipynb
├── training_text_title_comments_visual_preset_Add.ipynb
├── training_text_title_comments_visual_preset.ipynb
├── training_text_title_comments_visual_preset_Maximum.ipynb
├── training_text_title_meta_preset.ipynb
├── training_text_title_visual_meta_preset.ipynb
├── utils
│   ├── callbacks
│   │   ├── callbackUtils.py
│   │   ├── MyCallbacks.py
│   │   ├── MyTelegramCallBack.py
│   │   ├── MyTimeHistoryCallback.py
│   ├── datagenUtils
│   │   ├── datagenUtils.py
│   │   ├── DataSeqMetaModel.py
│   │   ├── DataSeqOneModel_Image.py
│   │   ├── DataSeqThreeModels_text_image_meta_old.py
│   │   ├── DateGenThreeModels.py
│   │   ├── dual_modal
│   │   │   ├── DataSeqImageTitle.py
│   │   │   ├── DataSeqMetaVisual.py
│   │   │   ├── DataSeqTitleComments.py
│   │   │   ├── DataSeqTitleMeta.py
│   │   ├── four_modal
│   │   │   ├── DataSeqFourModels.py
│   │   ├── three_modal
│   │   │   ├── DataSeqTitleCommentsMeta.py
│   │   │   ├── DataSeqTitleCommentsVisual.py
│   │   │   ├── DataSequenceImageCommentsMeta.py
│   │   │   ├── DataSequenceImageTextMeta.py
│   │   └── Untitled.ipynb
│   ├── fileDirUtils
│   │   ├── fileDirUtils.py
│   ├── image_models.py
│   ├── models
│   │   ├── modelUtils.py
│   ├── telegramUtils
│   │   └── telegram_bot.py
│   ├── text_processing.py
│   └── textUtils
│       ├── commentsProcessing.py
│       └── textpreprocessing.py
├── venv
```

## Download Google BERT Base Model
Please see [Readme Google Bert Models](https://github.com/google-research/bert/blob/master/README.md) for further information about the BERT Models.

For this repository please download: [Download BERT Model](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)

