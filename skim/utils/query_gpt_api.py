"""Usage:

python query_gpt.py \
--openai_api_key XXX \
--prompts_path XXX \
--engine gpt-4 \
--start_index 0 \
--end_index 1000 

Modify above command accordingly.
"""

import os
import json
from utils.openai_handler import OpenAIModelHandler
from tqdm import tqdm
import logging

import argparse

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.info("setting logging to info")

def main(openai_api_key, start_index, end_index, prompts_path, engine="gpt-4"):
    config = {
        "model": engine,
        "max_tokens": 200,
        "temperature": 0.8,
        "stop": ["<stop_tokens>"]
    }

    openai_api_config = {
        "openai_api_type": "", # NOTE: modify accordingly
        "openai_api_base": "", # NOTE: modify accordingly
        "openai_api_version": "", # NOTE: modify accordingly
    }

    handler = OpenAIModelHandler(
        config=config,
        openai_api_key=openai_api_key,
        timeout=5,
        max_attempts=10,
        **openai_api_config,
    )

    responses_folder = args.responses_folder
    if len(responses_folder) == 0:
        responses_folder = os.path.dirname(args.prompts_path)
        print("[INFO] No responses folder provided. Defaulting to:", responses_folder)
    else:
        print("[INFO] Responses dump folder provided:", responses_folder)
    
    print("[INFO] Loading prompts from:", args.prompts_path)

    all_prompts = json.load(open(prompts_path, "r"))
    # slice prompts
    all_prompts = all_prompts[start_index : end_index]
    # all_prompts = [x["prompt"] for x in all_prompts]

    print(f"Processing {len(all_prompts)} prompts...")
    print(f"From index {start_index} to {end_index}...")
    print()

    # Query GPT
    responses = handler.get_response(all_prompts)

    json.dump(
        [x[0] for x in responses], 
        open(os.path.join(responses_folder, f"responses_{start_index:06d}_to_{end_index:06d}.json"), "w"), 
        indent=4
    )

    print("Done...")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--openai_api_key", type=str, required=True, help="access token OpenAI")
    parser.add_argument("--start_index", type=int, default=0, help="start index to process")
    parser.add_argument("--end_index", type=int, default=None, help="end index to process")
    parser.add_argument("--prompts_path", type=str, required=True, help="path to the prompts json file")
    parser.add_argument("--responses_folder", type=str, default="", help="path to dump the gpt responses, defaults to the parent folder of the prompts_path")
    parser.add_argument("--engine", type=str, default="gpt-4", help="engine to infer on")

    args = parser.parse_args()
    print(args)
    main(args.openai_api_key, args.start_index, args.end_index, args.prompts_path, args.engine)