import setuptools

setuptools.setup(
    name="interface_master",
    version="1.1.1",
    install_requires=[
        "numpy",
        "pymatgen",
    ],
    author="Yaoshu Xie",
    author_email="yxie@iis.u-tokyo.ac.jp",
    description="A program of calculating interface structure",
    packages=setuptools.find_packages(),
    python_requires=">=3",
)
