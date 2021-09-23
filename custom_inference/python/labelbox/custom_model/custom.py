import sys

from labelbox import Client
from labelbox.schema.ontology import OntologyBuilder, Tool
from labelbox import Project, Dataset, Client, LabelingFrontend
from typing import Dict, Any
from flair.models import SequenceTagger
from flair.models import MultiTagger
from flair.data import Sentence
!pip install wikipedia-api
# !pip install wikipediaapi
import wikipediaapi
import uuid
import requests
import ndjson
import os
from getpass import getpass
import ipywidgets
from tqdm import tqdm
from ipywidgets import FloatProgress

import pandas as pd

API_KEY = pd.read_csv('mi_llave.secret').my_api.iloc[0]
ENDPOINT = "https://api.labelbox.com/graphql"
client = Client(API_KEY, ENDPOINT)


def load_model():
    modelo = MultiTagger.load(['pos'])
    return modedlo

def score(data, model, *):
    
    return model.predict(sentence)


if __name__== '__main__':
    ontology_builder = OntologyBuilder(
        tools=[Tool(tool=Tool.Type.NER, name="noun")]
    )

