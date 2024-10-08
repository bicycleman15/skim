# SKIM: Scalable Knowledge Infusion for Missing Labels

The repo contains the official code for the paper [On the Necessity of World Knowledge for Mitigating Missing Labels in Extreme Classification](https://arxiv.org/abs/2408.09585).

## Artifacts
We provide the following artifacts for future research and reproducibility:

1. SKIM augmented datasets that were used to train the XC models in the paper. One may directly use the provided `.npz` files to train on their favourite XC models/architectures. Available at: `artifacts/skim_augmented_datasets/<dataset>/trn_X_Y_skim.npz`.

2. Additionally, we provide all the required prompts, GPT4 responses, `litgpt` converted training format dataset for fintuning SLMs, fintuned SLMs, and the large-scale generated synthetic queries that can be used to obtain/reproduce the above `.npz` files. We provide detailed instructions and code on how one can obtain SKIM augmented dataset for their own XC datasets. These all can be found in `artifacts/` directory.

## Code

In order to obtain SKIM augmented datasets for your XC datasets, you can follow the steps outlined. 

On a high level, these would be (i) obtaining a SLM that can generate synthetic queries (this would require distilling this specific task using a much larger LLM e.g. GPT4 a few finetuning examples, (ii) generating large-scale synthetic queries using this finetuned SLM, (iii) mapping these synthetic queries to the train set queries using a pretrained XC encoder (e.g. NGAME in our case), and obtaining the final augmented dataset. Refer to below for more details:

Step 0: To perform task-specific distillation, refer to the diretory `skim/task-specific-distillation`. (Note that we call this step 0 since this would be the first thing one do when using SKIM. However, the paper does not talk about this step 0 explicitly.)

Step 1: To perform large-scale synthetic query generation, refer to the directory `skim/step-1`.

Step 2: Mapping synthetic queries to train set queries, refer to the directory `skim/step-2`.

You should now have a the SKIM augmented dataset in the form of `trn_X_Y_skim.npz`. Now train your favourite XC model/architecture on this SKIM augmented dataset.

## Requirements

Use the file `requirements.txt` to install the dependencies.

## Acknowledgements

We heavily rely on the following to train the XC models in our paper:
1. DEXML: https://github.com/nilesh2797/DEXML
2. Renee: https://github.com/microsoft/renee

## Issues

If you have any questions, feel free to open an issue on GitHub or contact the authors (Jatin Prakash (jatin.prakash@nyu.edu) or Anirudh Buvanesh (anirudh.buvanesh@mila.quebec)).

## Reference

If you find this repo useful, please consider citing:

```bibtex
@article{prakash2024necessity,
  title={On the Necessity of World Knowledge for Mitigating Missing Labels in Extreme Classification},
  author={Prakash, Jatin and Buvanesh, Anirudh and Santra, Bishal and Saini, Deepak and Yadav, Sachin and Jiao, Jian and Prabhu, Yashoteja and Sharma, Amit and Varma, Manik},
  journal={arXiv preprint arXiv:2402.05266},
  year={2024}
}
```