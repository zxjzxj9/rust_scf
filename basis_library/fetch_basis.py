#! /usr/bin/env python

import requests
import json
import pickle
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Optional
import curses
from curses import wrapper
import logging

def fetch_metadata():
    base_url = "https://www.basissetexchange.org"
    response = requests.get(base_url + '/api/metadata')
    return response.json()

def fetch_formats():
    base_url = "https://www.basissetexchange.org"
    response = requests.get(base_url + '/api/formats')
    return response.json()

def fetch_basis(basis_name: str, element: str, format='nwchem'):
    base_url = "https://www.basissetexchange.org"
    response = requests.get(base_url + f'/api/basis/{basis_name}/format/{format}?elements={element}')
    # return response.json()
    if format == 'json':
        return response.json()
    else:
        return response.text


if __name__ == "__main__":
    # metadata = fetch_metadata()
    # formats = fetch_formats()
    # print("Metadata:", metadata)
    # print("Formats:", formats)

    basis_name = "6-31g"
    element = "H"
    basis = fetch_basis(basis_name, element)
    print("Basis:\n", basis)