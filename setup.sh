#!/usr/bin/env bash

parent_path=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )
cd "$parent_path"

ant -buildfile lib/cistern/marmot/build.xml
python3 -m pip install -r requirements.txt
