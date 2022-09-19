import pathlib
from setuptools import setup

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="honest_ml",
    version="0.9.0",
    description="Training for models conforming to the scikit-learn api",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/EricSchles/honest_ml",
    author="Eric Schles",
    author_email="ericschles@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    packages=[
        "honest_ml",
        'honest_ml.trainer',
        'honest_ml.visualizer',
        'honest_ml.experiment',
        'honest_ml.utils'
    ],
    include_package_data=True,
    install_requires=["sklearn", "scipy", "numpy",
                      "statsmodels", "mlxtend", "pytest", "seaborn"],
)
