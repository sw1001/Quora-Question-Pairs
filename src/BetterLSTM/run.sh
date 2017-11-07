#!/usr/bin/env bash

python3 non_nlp_feature_extraction.py
python3 nlp_feature_extraction.py
python3 model.py
python3 postprocess.py