# Interface master

[![GitHub Pages](https://github.com/nmdl-mizo/interface_master/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/nmdl-mizo/interface_master/actions/workflows/gh-pages.yml)

A python package building CSL or approximate CSL interfaces of any two lattices and computing cell of non-identical displacement (CNID).

## Functions
1. Searching for 3D and 2D CSLs by rotating along an axis;
2. Build an approximate CSL interface according to a pre-determined approximate disorientation;
3. Generate exact CSL interfaces;
4. Sampling Rigid Body Translation, aka, RBT in CNID

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

## Automatically generate heterogeneous interfaces without repeating identical ones; and ensure to get minimum-sized supercell, and giving information of the rational coordinates
![截屏2024-03-14 10 26 56](https://github.com/nmdl-mizo/interface_master/assets/48645456/d0986de7-ec1a-4e3b-b828-31b87c355f00)

![image](https://github.com/nmdl-mizo/interface_master/assets/48645456/cc6da36c-92ee-4e30-a64b-43dee69d166f)

## Requirements
- python3
- pymatgen
- matplotlib
- gb_code
- numpy (requirements already satisfied by installing pymatgen)

## Documentation

[Documentation including tutorials](https://nmdl-mizo.github.io/interface_master/) is available (only for development branch).

Tutorials are ready on [Google Colab](https://colab.research.google.com/github/nmdl-mizo/interface_master/blob/develop).

## How to cite
 If you use the interface_master, please cite the following articles [1][2].  
[1] "interface_master: Python package building CSL and approximate CSL interfaces of any two lattices -- an effective tool for interface engineers"  
 Y. S. Xie, K. Shibata, and T. Mizoguchi, ArXiv:2211.15173 (https://arxiv.org/abs/2211.15173). 

[2] "A brute-force code searching for cell of non-identical displacement for CSL grain boundaries and interfaces"  
 YS. Xie, K. Shibata, and T. Mizoguchi  
 Comp. Phys. Comm. 273 (2022) 108260-1-8 (https://www.sciencedirect.com/science/article/pii/S0010465521003726)
