# HVAC Gym

<<<<<<< HEAD
HVAC Gym is a data-driven modelling and reinforcement learning environment for building HVAC systems.
=======
Automated modelling of building HVAC and creation of reinforcement learning gym
environments

## Authentication

Installing currently requires access to some dependencies hosted on the
private `csiroenergy` Azure package repository.
Please floow steps
in [Azure Package Repo Setup](https://confluence.csiro.au/display/GEES/Poetry+Cheat+Sheet#PoetryCheatSheet-InstallFromandPublishtoourPrivatePyPiindex)
to obtain and set an API key before proceeding.

You'll also need to add your Senaps and DCH API keys to a .env file in the project root:

```shell
cd hvac_gym
echo SENAPS_API_KEY=XXXXXXX >> .env
echo DCH_API_KEY=XXXXXXX >> .env
```
You'll also need to request access to the CSIRO buildings in Data Clearing House (DCH). Talk to Matt Amos or Akram Hameed for access.
>>>>>>> temp-branch

It includes methods of creating a data-driven model of whole building HVAC operation, trained on real data from building
management systems and electrical metering. These models are then packaged into a 'Gym' environment, which can be used
to train and test control systems (such as Reinforcement Learning agents or Model Predictive Control methods) to control
the HVAC system model.

<<<<<<< HEAD
The ultimate goal is to allow quick and repeatable automated building modelling on new buildings, allowing novel
controllers to be trained/configured before deploying them to control real buildings.

The model training is designed to be sourced from the CSIRO [Data Clearing House (DCH)](http://dataclearinghouse.org),
with points discovered via [Brick Schema](https://brickschema.org) building models, or for users to provide their own
data in the correct tabular format (coming soon). A set of sample building data and models are provided in the current
release.

:warning: The current release is a proof-of-concept. Additional features are planned that may introduce breaking
changes.

# Usage

This repository can be used as-is with the included sample data and models for simulating a building's HVAC system and
controlling it with a simple example agent. This agent can be used as the basis for writing and testing more complex
controllers.

## Installation

To run this repository, you'll need to
install [git](https://git-scm.com/downloads), [python](https://www.python.org/downloads/) >= 3.12,
and [poetry](https://python-poetry.org/docs/#installation) >= v1.8, and then run the following commands:

```shell
git clone git@github.com:csiro-energy-systems/hvac_gym.git # To set up SSH keys, see https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh
git clone https://github.com/csiro-energy-systems/hvac_gym.git # or clone via https
cd hvac_gym
poetry env use path/to/python>=3.12 # replace with your python executable's path
poetry install
poetry run poe clean # clean any old cache files etc
poetry run poe unit_tests # optional: run unit tests
=======
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
poetry env use path/to/python>3.9.exe
poetry install

# Run poe targets for cleaning, training, and running the gym.  See [tool.poe.tasks] section in pyproject.toml for full commands.
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

### Tools

If you don't already have them, install some basic dev tools

#### Pyenv

Pyenv is a tool for managing multiple python versions on the same machine.

```shell
# Linux & WSL:
curl https://pyenv.run | bash # Afterwards, follow print instructions to update PATH in your ~/.bashrc and restart your shell.

# Mac:
brew update; brew install pyenv

# Windows Admin Powershell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"

# All platforms:
pyenv install --list # list available python versions
pyenv install 3.x.x # install preferred python version
```

#### Poetry

There ae several ways to install poetry, but the pipx method is recommended.
See [here](https://python-poetry.org/docs/#installation) for more details.

```shell
pyenv shell 3.x.x # start a shell for any
python -m pip install pipx # install pipx
pipx install poetry # use pipx to install poetry in a dedicated virtual env and add it to your PATH
```

### Project

If cloning this repo, and the tools above are installed, run these commands to set up
the environment for development:

```shell
# Clone the repo
git clone <this-repo.git>
cd  <this-repo-url>
>>>>>>> temp-branch

# Get test data from LFS (large file system)
git lfs install # windows
apt install git-lfs # linux
git lfs pull # pull data for current branch

unzip data/sample-models.zip # or manually extract sample models and data into hvac_gym/output/
ls output # check that output/ now contains three .pkl files and a .parquet file.
poetry run poe example  # run simple example agent against gym
```

<<<<<<< HEAD
If you'd like to implement your own control agent, see [hvac_agents.py](src/hvac_gym/gym/hvac_agents.py)
and [test_gym.py](src/tests/test_gym.py) for simple examples of an agent and the example gym runner respectively.

# Gym Environment

The current gym runs models the whole-building behaviour of a 3-storey office building with in NSW, Australia.
Individual AHU and zone measurements are available for the building, but to simplify the model (for speed and
troubleshooting purposes), the median of each sensor and setpoint type is taken across all zones, and the HVAC system
modelled as a single large zone.

In short, this is a lumped model which predicts the whole-building median zone temperature and chiller power for a
sequence of the actions below.

The gym's inputs and outputs are described below.

- Actions:
    - Chilled water valve position (0-100%)
    - Hot water valve position (0-100%)
        - :warning: Note: Heating mode is not yet implemented, so this action has no effect yet.
    - Supply air fan speed (0-100%)
    - Outside air damper position (0-100%)
- Observations
    - zone temperature (°C)
    - chiller power (kW)
    - ambient zone temperature (°C)
        - This is an intermediate output, predicting the zone temperature as if no mechanical cooling/heating was
          applied.
- Actuals:
    - Actual zone temperature (°C): the real building's median zone temperature, as was measured by physical sensors.
    - Actual chiller power (kW): the real building's total chiller power, as was measured by electrical power meters.
- Reward
    - A reward function is passed into the to allow users to customise their agent's reward function. It can be
      calculated from any of the observations and actions.

# Developing

:warning: Note: Full development of this library (for training new models etc) currently requires some access to some
non-public software infrastructure, sois only possible for CSIRO staff right now. Some of the links and authentication
below won't work for the general public. We hope to improve this in later releases.

## Authentication

Installing currently requires access to some dependencies hosted on the private `csiroenergy` Azure package repository.
Please follow steps
in [Azure Package Repo Setup](https://confluence.csiro.au/display/GEES/Poetry+Cheat+Sheet#PoetryCheatSheet-InstallFromandPublishtoourPrivatePyPiindex)
to obtain and set an API key before proceeding.

You'll also need to add your Senaps and DCH API keys to a .env file in the project root:
=======
Other useful commands include:
>>>>>>> temp-branch

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
