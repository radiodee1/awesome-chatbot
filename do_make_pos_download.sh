#!/bin/bash

cd raw

#browse https://www.kaggle.com/abhinavwalia95/entity-annotated-corpus#ner_dataset.csv

kaggle datasets download -d abhinavwalia95/entity-annotated-corpus

unzip entity-annotated-corpus.zip