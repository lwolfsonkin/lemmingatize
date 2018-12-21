#!/usr/bin/env bash

ant -buildfile lib/cistern/marmot/build.xml
python3 -m pip install -r requirements.txt
