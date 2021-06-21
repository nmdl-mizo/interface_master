import setuptools


def _requires_from_file(filename):
    return open(filename).read().splitlines()


setuptools.setup(
    name="interfacemaster",
    version="1.1.1",
    install_requires=_requires_from_file('requirements.txt'),
    author="Yaoshu Xie",
    author_email="yxie@iis.u-tokyo.ac.jp",
    description="A program of calculating interface structure",
    packages=setuptools.find_packages(),
    python_requires=">=3",
)
