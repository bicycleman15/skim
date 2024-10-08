# Step 2: Mapping synthetic queries to train set queries

Now we need to map synthetic queries to actual queries in our dataset. For this, we use a pretrained XC encoder (e.g. NGAME in our case) to perform this. We can use the following steps to acheive this:

1. We first compute embeddings for our existing train queries. We then compute embeddings for the generated synthetic queries. We can use the file present at `skim/step-2/create_embeddings.py` for this. We provide the pretrained NGAME encoder used in our experiments for this in the directory: `artifacts/pretrained-models`.

> Note that to compute embeddings for the synthetic queries, we would first need to tokenize those. We can use `skim/step-2/utils/CreateTokenizedFiles.py` for this.

2. Then perform nearest neigbour search for every synthetic query to the closest actual queries in the NGAME encoder embedding space. This can be done using `skim/step-2/gpu_anns.py`.

3. Finally, make the final training matrix `trn_X_Y_skim.npz` to train the XC models on. To do this, we can use the respective notebooks `skim/step-2/create_npz_orcas.ipynb` or `skim/step-2/create_npz_wiki.ipynb` (you may need to modify these to run on your custom XC datasets). We provide this training matrix for experiments done in the paper for future research and reproducibility at `artifacts/skim-augmented-datasets`. 

4. That's it, now you are ready to train XC models. Just swap the old `trn_X_Y.npz` with the new `trn_X_Y_skim.npz` in your favourite XC algorithm implementations and you are set! In the paper, we use official implementations of DEXML (https://github.com/nilesh2797/DEXML) and Renee (https://github.com/microsoft/renee) to train XC models on the augmented SKIM dataset (i.e. `trn_X_Y_skim.npz`).