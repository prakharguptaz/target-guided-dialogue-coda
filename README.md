# Target-Guided Dialogue Response Generation Using Commonsense and Data Augmentation

Code for paper Target-Guided Dialogue Response Generation Using Commonsense and Data Augmentation- [Link](https://arxiv.org/abs/2205.09314) (NAACL 2022 Findings paper)

## Overview
Target-guided response generation enables dialogue systems to smoothly transition a conversation from a dialogue context toward a target sentence. Such control is useful for designing dialogue systems that direct a conversation toward specific goals, such as creating non-obtrusive recommendations or introducing new topics in the conversation. In this paper, we introduce a new technique for target-guided response generation, which first finds a bridging path of commonsense knowledge concepts between the source and the target, and then uses the identified bridging path to generate transition responses. Additionally, we propose techniques to re-purpose existing dialogue datasets for target-guided generation. Experiments reveal that the proposed techniques outperform various baselines on this task. Finally, we observe that the existing automated metrics for this task correlate poorly with human judgement ratings. We propose a novel evaluation metric that we demonstrate is more reliable for target-guided response evaluation. Our work generally enables dialogue system designers to exercise more control over the conversations that their systems produce.

### Data and model files
Please find data and trained models for this paper in this [Google drive folder](https://drive.google.com/drive/folders/1xuzHVqoLXUkD9XCjWpB5zFsJlBvKetAT?usp=sharing)



## Commonsense model
Scripts present in folder scripts/commonsense_models

## Data augmentation
Scripts for data augmentation using Dailydiaog dataset present in folder scripts/data_augmentation

## Targetcoherence metric
Scripts present in folder scripts/targetcoherence_metric

## Path generation
Scripts present in folder scripts/path_generation

## Response generation
Scripts present in folder scripts/response_generation

<!-- ## Citation

Please cite our paper with the following bibtex:

```bibtex
  title={Target-Guided Dialogue Response Generation Using Commonsense and Data Augmentation},
  author={Gupta, Prakhar and Jhamtani, Harsh and Bigham, Jeffrey P},
  journal={arXiv preprint arXiv:2205.09314},
  year={2022}
}
``` -->
