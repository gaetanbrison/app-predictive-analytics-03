# Import required functions
from setuptools import setup, find_packages, Command
import pathlib
import os


# The directory containing this file
HERE = pathlib.Path(__file__).parent



# Call setup function
setup(
    author=("Gaëtan Brison (ML Research Engineer)"),
    description="App Predictive Analytics",
    name="IOKR-Test",
    version="0.2.1",
    license="MIT",
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    test_suite="nose.collector",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3",
    ],
    install_requires=["pandas", "numpy", "scipy", "scikit-learn","prophet","fbprophet","htbuilder","streamlit","altair"],
    python_requires=">=3.7",
)


class CleanCommand(Command):
    """Custom clean command to tidy up the project root."""

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system("rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info")


# Further down when you call setup()
setup(
    # ... Other setup options
    cmdclass={
        "clean": CleanCommand,
    }
)