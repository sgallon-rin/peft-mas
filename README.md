# PEFT-MAS

<p align="left">
    <a href="https://img.shields.io/badge/PRs-Welcome-red">
        <img src="https://img.shields.io/badge/PRs-Welcome-red">
    </a>
    <a href="https://img.shields.io/github/last-commit/HKUNLP/UnifiedSKG?color=green">
        <img src="https://img.shields.io/github/last-commit/sgallon-rin/peft-mas?color=green">
    </a>
    <br/>
</p>

This repository contains source code for NLPCC 2024 paper: 
Leveraging Parameter-Efficient Fine-Tuning for Multilingual Abstractive Summarization


## Environment

python version: 3.9.13

```shell
$ conda create -n peft-mas python=3.9.13
$ conda activate peft-mas
$ pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install transformers==4.9.2
$ pip install datasets==2.19.2
$ pip install sentencepiece==0.2.0
```

### Install Required Packages for Evaluation

#### Multilingual ROUGE score

Use (language-specific) Multilingual ROUGE score instead of the original 
`rouge_score` package (which is designed for English) 
to avoid tokenizing/stemming problems.

See: https://github.com/csebuetnlp/xl-sum/tree/master/multilingual_rouge_scoring

```shell
$ git clone git@github.com:csebuetnlp/xl-sum.git
$ cd xl-sum
$ git checkout 6b97e97 # api has been changed after this commit
$ cd multilingual_rouge_scoring
$ pip3 install -r requirements.txt
$ python3 -m unidic download # used for japanese segmentation
$ pip3 install --upgrade ./
```

#### Download NLTK resource

```python
>>> import nltk
>>> nltk.download('punkt_tab')
```

### Set data and output directory (Optional)

```shell
$ ln -s /path/to/data/ data
$ ln -s /path/to/output output
```

## Training and Evaluation

```shell
# Common
$ export LANG=arabic  # options: [arabic, burmese, chinese_simplified, english, french, hindi, japanese, korean, russian, spanish, turkish]
$ export SEED=123  # random seed
$ export LR=5e-5  # learning rate
$ export EPOCH=3  # training epoch

# (1) Compare fine-tuning, prefix-tuning(l=100), lora(r=16)
$ export MODEL=mBART_large_50 # options: [mBART_large_50, mT5_base]
$ export TUNE=finetune  # options: [finetune, lora, prefix]
$ python train.py --seed ${SEED} --cfg xlsum/${MODEL}/${TUNE}/${LANG}.cfg --run_name ${MODEL}_${TUNE}_${LANG} --logging_strategy steps --logging_first_step true --logging_steps 1000 --evaluation_strategy steps --eval_steps 100000 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 100000 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs ${EPOCH} --adafactor true --learning_rate ${LR} --do_train --do_eval --do_predict --predict_with_generate --output_dir output/${MODEL}_${TUNE}_${LANG}_seed${SEED}_lr${LR} --overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --generation_num_beams 4 --generation_max_length 60 --input_max_length 1024 --ddp_find_unused_parameters true

# (2) Prefix-tuning, compare different prefix lengths (mBART only)
$ export MODEL=mBART_large_50
$ export TUNE=finetune  # options: [finetune, prefix]
$ export PREFIXLEN=200  # prefix length, options: [30, 50, 200, 300, 400]
$ python train.py --seed ${SEED} --cfg xlsum/${MODEL}/${TUNE}/${PREFIXLEN}/${LANG}.cfg --run_name ${MODEL}_${TUNE}${PREFIXLEN}_${LANG} --logging_strategy steps --logging_first_step true --logging_steps 1000 --evaluation_strategy steps --eval_steps 100000 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 100000 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs ${EPOCH} --adafactor true --learning_rate ${LR} --do_train --do_eval --do_predict --predict_with_generate --output_dir output/${MODEL}_${TUNE}${PREFIXLEN}_${LANG}_seed${SEED}_lr${LR} --overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --generation_num_beams 4 --generation_max_length 60 --input_max_length 1024 --ddp_find_unused_parameters true

# (3) Few-shot (mBART only)
$ export MODEL=mBART_large_50
$ export TUNE=finetune  # options: [finetune, prefix]
$ python train.py --seed ${SEED} --cfg few_shot_xlsum/${MODEL}/${TUNE}/${LANG}.cfg --run_name ${MODEL}_${TUNE}_${LANG} --logging_strategy steps --logging_first_step true --logging_steps 1000 --evaluation_strategy steps --eval_steps 100000 --metric_for_best_model avr --greater_is_better true --save_strategy steps --save_steps 100000 --save_total_limit 1 --load_best_model_at_end --gradient_accumulation_steps 4 --num_train_epochs ${EPOCH} --adafactor true --learning_rate ${LR} --do_train --do_eval --do_predict --predict_with_generate --output_dir output/few_shot_${MODEL}_${TUNE}_${LANG}_seed${SEED}_lr${LR} --overwrite_output_dir --per_device_train_batch_size 8 --per_device_eval_batch_size 8 --generation_num_beams 4 --generation_max_length 60 --input_max_length 1024 --ddp_find_unused_parameters true
```

For more examples of commands, please refer to [run_commands.txt](./run_commands.txt).

For detailed configurations, please refer to [config](./config).


## Code Structure Overview
    .
    ├── config                                 # Config files for experiments, tasks, and settings
    │   ├── META_TUNING                        # Config files for tasks and settings
    │   ├── xlsum                              # Config files for experiments (xlsum)
    │   └── few_shot_xlsum                     # Config files for experiments (few-shot xlsum)
    │
    ├── metrics                                # Code for evaluation
    │   └── ...                                # Please check the README of the ./seq2seq_construction.
    │
    ├── models                                 # Code for models
    │   ├── peft                               # Code for T5 and BART with LoRA (based on HuggingFace PEFT)
    │   ├── prompt                             # Code for T5 and BART with prefix-tuning (based on HuggingFace Transformers)
    │   └── unified
    │           ├── base.py                    # Code for the base model that enables an arbitrary model to be pushed to HuggingFace Model Hub (namely, PushToHubFriendlyModel)
    │           ├── finetune.py                # Code for finetuning
    │           ├── peft.py                    # Code for peft (lora)
    │           └── prefixtuning.py            # Code for prefix-tuning
    │
    ├── seq2seq_construction                   # Code for converting raw data into sequences
    │    └──  ...                              # Please check the README in this directory.
    │
    ├── tasks                                  # Code for loading raw data
    │    └──  ...                              # Please check the README in this directory.
    │
    ├── utils                                  # Code for utils
    │    ├── config.py                         # Code for parsing config files in ./config
    │    ├── dataset.py                        # Code for converting input and output sequences into Datasets for training
    │    ├── tool.py                           # Code for loading models, seq2seq constructors, and evaluators
    │    ├── trainer.py                        # Code for EvaluationFriendlyTrainer. If you want make training-specific modifications, you may want to change something here.
    │    └── training_arguments.py             # Code for seq2seq training arguments
    │
    ├── .gitignore
    ├── environment.yml                        # Conda environment config file
    ├── README.md                              # README
    ├── run_commands.txt                       # Run commands examples
    └── train.py                               # Entry code, which controls train, eval, test, storage, and logging


## References

- https://github.com/XiangLi1999/PrefixTuning
- https://github.com/xlang-ai/UnifiedSKG
- https://github.com/csebuetnlp/xl-sum


[//]: # (## Citation)

[//]: # (If you find our work helpful, please cite as:)

[//]: # (```)
[//]: # (TODO)
[//]: # (```)
