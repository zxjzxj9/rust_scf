# Basis Library

---
The basis library download from https://www.basissetexchange.org/

The basis library is a collection of electronic structure basis sets in a common format, which is used by many electronic structure programs. The basis library is a project of the Molecular Sciences Software Institute (MolSSI).

## Installation
We use PyO3 to bind Rust and Python. You can install the basis library by running the following command:

```bash
maturin develop 
```

or 

```bash
maturin publish
```

To install the rust dependency for Python