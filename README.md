# hvac_gym

HVAC Gym is a data-driven modelling and reinforcement learning environment for building HVAC systems.

It includes methods of creating a data-driven model of whole building HVAC operation, trained on real data from building
management systems and electrical metering. These models are then packaged into a 'Gym' environment, which can be used
to train and test control systems (such as Reinforcement Learning agents or Model Predictive Control methods) to control
the HVAC system model.

The ultimate goal is to allow quick and repeatable automated building modelling on new buildings, allowing novel
controllers to be trained/configured before deploying them to control real buildings.

The model training is designed to be sourced from the CSIRO [Data Clearing House (DCH)](http://dataclearinghouse.org),
with points discovered via [Brick Schema](https://brickschema.org) building models, or for users to provide their own
data in the correct tabular format (coming soon). A set of sample building data and models are provided in the current
release.

# Usage

This repository can be used as-is with the included sample data and models for simulating a building's HVAC system and
controlling it with a simple example agent. This agent can be used as the basis for writing and testing more complex
controllers.

## Installation

To run this repository, you'll need to
install [git](https://git-scm.com/downloads), [python](https://www.python.org/downloads/),
and [poetry](https://python-poetry.org/docs/#installation), and then run the following commands:

```shell
git clone git@github.com:csiro-energy-systems/hvac_gym.git
cd hvac_gym
poetry env use path/to/python>=3.11 # replace with your python executable's path
poetry install
poetry run poe clean # clean any old cache files etc
poetry run poe unit_tests # optional: run unit tests

# Get test data from LFS (large file system)
git lfs install # windows
apt install git-lfs # linux
git lfs pull # pull data for current branch

unzip data/sample-models.zip # extract sample models and data into output/
poetry run poe example  # run simple example agent against gym
```
If you'd like to implement your own control agent, see [hvac_agents.py](src/hvac_gym/gym/hvac_agents.py) and [test_gym.py](src/tests/test_gym.py) for simple examples of an agent and the example gym runner respectively.

# Developing

:warning: Note: Full development of this library currently requires some access to various software infrastructure, so
is only possible for CSIRO staff right now. Some of the links and authentication below won't work for the
general public. We hope to improve this in later releases.

## Authentication

Installing currently requires access to some dependencies hosted on the private `csiroenergy` Azure package repository.
Please floow steps
in [Azure Package Repo Setup](https://confluence.csiro.au/display/GEES/Poetry+Cheat+Sheet#PoetryCheatSheet-InstallFromandPublishtoourPrivatePyPiindex)
to obtain and set an API key before proceeding.

You'll also need to add your Senaps and DCH API keys to a .env file in the project root:

```shell
cd hvac_gym
echo SENAPS_API_KEY=XXXXXXX >> .env
echo DCH_API_KEY=XXXXXXX >> .env
```

You'll also need to request access to the CSIRO buildings in Data Clearing House (DCH). Talk to Matt Amos or Akram
Hameed for access.

Check the [Authentication](#Authentication) section above, then run the following
commands to install the packages and run the gym:

```shell
# Check your poetry version.  You'll need >=1.8.x.
poetry --version

# Set up Azure package authentication (see Authentication section above)
poetry source add pypi
poetry source add --priority supplemental csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/simple/
poetry config repositories.csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/upload
poetry config http-basic.csiroenergy IDENT TOKEN # Note: replace IDENT and TOKEN with your Azure credentials (see above)

# Install packages
poetry env use path/to/python>=3.11.exe # replace with your python executable's path
poetry install --all-extras

# Run poe targets for cleaning, training, and running the gym.  See [tool.poe.tasks] section in pyproject.toml for full commands.
poetry run poe init # (re)set up the environment after a clone or pull
poetry run poe clean # clean any old cache files, outputs etc
poetry run poe train  # train models
poetry run poe example  # run simple example agent against gym
```

To implement your own agent, see the
simple [MinMaxCoolAgent](src/hvac_gym/gym/hvac_agents.py) example agent and
the [TestGym](src/tests/test_gym.py) test.
If you don't intend to modify this code, we recommend installing this package as a
dependency of your own project, rather than cloning this repo, e.g.:

```shell
cd your_project
poetry add git+ssh://github.com/csiro-energy-systems/hvac_gym.git # assuming you have github ssh keys set up
```

## Support / Contributions

Github issue reports and pull requests are welcome. The authors will endeavour to respond in a timely manner.
