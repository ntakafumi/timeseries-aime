# setup.py
from setuptools import setup, find_packages

setup(
    name="tsaime",
    version="0.1.0",
    description="Time-Series AIME: Rolling AIME on top of aime-xai and pyEDM",
    author="Takafumi Nakanishi",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "pandas>=1.3",
        "aime-xai>=0.1.0",   
        "pyEDM>=1.14.0.2"
    ],
    license="Academic and Non-Commercial Research License:Commercial use is prohibited unless a separate written commercial license is obtained from the copyright holder.",
)