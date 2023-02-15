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

## Documentation

[Documentation including tutorials](https://nmdl-mizo.github.io/interface_master/) is available (only for development branch).

## How to cite
 If you use the interface_master, please cite the following articles [1][2].  
[1] "interface_master: Python package building CSL and approximate CSL interfaces of any two lattices -- an effective tool for interface engineers"  
 Y. S. Xie, K. Shibata, and T. Mizoguchi, ArXiv:2211.15173 (https://arxiv.org/abs/2211.15173). 

[2] "A brute-force code searching for cell of non-identical displacement for CSL grain boundaries and interfaces"  
 YS. Xie, K. Shibata, and T. Mizoguchi  
 Comp. Phys. Comm. 273 (2022) 108260-1-8 (https://www.sciencedirect.com/science/article/pii/S0010465521003726)
