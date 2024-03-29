import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="randomizer_ml",
    version="0.11.2",
    description="Training for models conforming to the scikit-learn api",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/EricSchles/randomizer_ml",
    author="Eric Schles",
    author_email="ericschles@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10"
    ],
    packages=[
        "randomizer_ml",
        'randomizer_ml.trainer',
        'randomizer_ml.visualizer',
        'randomizer_ml.experiment',
        'randomizer_ml.utils'
    ],
    include_package_data=True,
    install_requires=["scikit-learn", "scipy", "numpy",
                      "statsmodels", "mlxtend", "pytest", "seaborn"],
)
