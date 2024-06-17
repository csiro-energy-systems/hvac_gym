# hvac_gym

Automated modelling of building HVAC and creation of reinforcement learning gym environments

## Authentication

Installing currently requires access to some dependencies hosted on the private csiroenergy Azure package repository.
See [Azure Package Repo Setup](https://confluence.csiro.au/display/GEES/Poetry+Cheat+Sheet#PoetryCheatSheet-InstallFromandPublishtoourPrivatePyPiindex)

## Quick start:

Check the [Authentication](#Authentication) section above, then run the following commands to install the packages and run the gym:

```shell
poetry env use path/to/python>3.9.exe
poetry install

poetry shell
poe clean # clean any old cache files, outputs etc
poe train  # train models
poe example  # run simple example agent against gym
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
There ae several ways to install poetry, but the pipx method is recommended. See [here](https://python-poetry.org/docs/#installation) for more details.
```shell
pyenv shell 3.x.x # start a shell for any
python -m pip install pipx # install pipx
pipx install poetry # use pipx to install poetry in a dedicated virtual env and add it to your PATH
```

### Project

If cloning this repo, and the tools above are installed, run these commands to set up the environment for development:
```shell
# Clone the repo
git clone <this-repo.git>
cd  <this-repo-url>

# Get test data from LFS (large file system)
git lfs install # windows
apt-get install git-lfs # linux
git lfs fetch --all # fetch data from remote for all branches
git lfs pull # pull data for current branch

# We recommend using `pyenv` to manage multiple python versions. If pyenv is installed, just run:
pyenv install 3.x.y # install preferred python version
pyenv local 3.x.y # set that version as the default for this dir
pyenv versions # view currently installed python versions

# ...OR if you don't use pyenv, or you do but are on windows, run:
poetry env use c:\path\to\python3.9\python.exe # Windows
poetry env use /path/to/python39 # Linux/Mac

# install libraries
poetry install

# install pre-commit hooks for code quality etc
pre-commit install

# run tests locally
pytest

```
Other useful commands include:
```shell

# generate and show html doc
poetry run poe doc
poetry run poe show_doc

# Add private pypi index for adding and publishing packages
# For poetry >= 1.5
poetry source add pypi
poetry source add --priority supplemental csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/simple/
poetry config repositories.csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/upload
poetry config http-basic.csiroenergy YOUR_IDENT YOUR_TOKEN # Generate a token at https://dev.azure.com/csiro-energy/_usersSettings/tokens

# build a redistributable python package
poetry build

# publish a built wheel to the private csiroenergy pypi repo. Only works once for each unique project version (in pyproject.toml).
poetry publish -r csiroenergy

# build a docker image, and define two containers, one for tests, and one for launching your code
docker-compose build

# run tests in docker container
docker-compose run test

# To trigger a github release, increment the pyproject version create and push a matching git tag.
# E.g. to release version 0.0.2a ('n.n.n' creates a release, anything else creates a pre-release'):
poetry version 0.0.2a  # pre-release
git tag -a 0.0.2a -m "your tag description here"; git push --tags

```

## Installation Instructions

For detailed installation instructions see the template readme [here](docs/README.md)
