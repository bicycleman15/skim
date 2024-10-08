# Step 1: Generating diverse synthetic queries

Run the finetuned SLM on all the train queries (and their associated metadata) present in the dataset. You can use the file `large_scale_inference.py` for this. A sample command to run is provided in the file itself.

We provide the synthetic queries that were generated in our experiments using the finetuned SLM for the paper at: `artifacts/slm-generations`.

Proceed to Step-2 now to create the SKIM augmented dataset to finally train XC models on.