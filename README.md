# Interface master
A python package building CSL or approximate CSL interfaces of any two lattices and computing cell of non-identical displacement (CNID).

## Functions
1. Searching for 3D and 2D approximate CSL of any two lattices input by two cif files;
2. Generating CSL interfaces;
3. Computing DSC, CSL, CNID;
4. Searching for different terminating planes and visualizing them;

## Installation
```bash
pip install git+https://github.com/nmdl-mizo/interface_master
```
or
```bash
git clone https://github.com/nmdl-mizo/interface_master
cd interface_master
pip install .
```

## Requirements
- python3
- pymatgen
- matplotlib
- gb_code
- numpy (requirements already satisfied by installing pymatgen)
