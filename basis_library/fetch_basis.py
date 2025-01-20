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
from basis_py import load_basis_from_py_str

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
    # element = "H"
    element = "O"
    basis = fetch_basis(basis_name, element)
    print("Basis:\n", basis)
    load_basis_from_py_str(basis, "{}.{}.pkl".format(element, basis_name))

    # prettify print the dictionary
    with open("{}.{}.pkl".format(element, basis_name), 'rb') as f:
        res = pickle.load(f)
        print(len(res['basis_set'])) # size 2, correct
        for basis in res['basis_set']:
            print(len(basis))
        print(list(res['basis_set']))
        # print(json.dumps(res, indent=4))