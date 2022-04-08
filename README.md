# Interface master
A python package building CSL or approximate CSL interfaces of any two lattices and computing cell of non-identical displacement (CNID).

## Functions
1. Searching for 3D and 2D CSLs by rotating along an axis;
2. Build an approximate CSL interface according to a pre-determined approximate disorientation;
3. Generate exact CSL interfaces;
4. Sampling RBT in CNID

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
