#!/bin/bash
python main.py config.ini \
    --base-dir input_data/images_dreambooth \
    --output-dir output \
    --concepts input_data/concepts_dreambooth.txt \
    --prompts input_data/prompts.txt
