<h2 align="center"><b><h3>Not all layers are equally as important:</h3><h3>Every Layer Counts BERT</h3></b></h2><br>


<p align="center">
  <b>Lucas Georges Gabriel Charpentier and David Samuel</b>
</p>

<p align="center">
  <i>
    University of Oslo<br>
    Language Technology Group<br>
  </i>
</p>
<br>

<p align="center">
  <a href="https://aclanthology.org/2023.conll-babylm.20/"><b>Paper</b></a><br>
  <a href="https://huggingface.co/lgcharpe/ELC_BERT_baby_100M"><b>HuggingFace 100M model</b></a><br>
  <a href="https://huggingface.co/lgcharpe/ELC_BERT_small_baby_10M"><b>HuggingFace 10M model</b></a>
</p>

_______

<br>

<h3 align="center"><b>Abstract</b></h3><br>

This paper introduces a novel modification of
the transformer architecture, tailored for the
data-efficient pretraining of language models.
This aspect is evaluated by participating in the
BabyLM challenge, where our solution won
both the STRICT and STRICT-SMALL tracks.
Our approach allows each transformer layer to
select which outputs of previous layers to pro-
cess. The empirical results verify the potential
of this simple modification and show that not
all layers are equally as important.

_______

<br>

This is the official repository for our BabyLM 2023 submission: ELC-BERT.

_______

<br>

## Content of this repository

- `./train_elc_bert_*.py`: Scripts to train an ELC-BERT model (replace * with base, normalized, weighted_output, or zero).
	- base: LTG-BERT (https://aclanthology.org/2023.findings-eacl.146.pdf)
	- zero: All layer weights initialised as zeros
	- normalized: Same as above, with vector representation normalised to a unit vector
	- weighted_output: Same as zero initialisation, and input to LM head is a weighted sum of all layers
- `./configs/`: Folder containing model configs.
- `./pre_training/`: Scripts for the dataset, optimizer and utilities of pretraining.
- `./models/`: Folder containing training models.
- `./checkpoints/`: Folder containing pre-trained models.
- `./trained_models/`: Folder containing fine-tuned models.
- `./logs/`: Folder containing fine-tuning logs.


