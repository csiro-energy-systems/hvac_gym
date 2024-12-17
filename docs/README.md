# Example Python project template

[![Github Unit Test Status](https://github.com/csiro-energy-systems/_PythonTemplate/actions/workflows/poetry-tests.yml/badge.svg)](https://github.com/csiro-energy-systems/_PythonTemplate/actions/workflows/poetry-tests.yml)

This is an example of what a typical Python project should look like, and is recommended as the basis of most of our
python projects.

If you know what you're doing, try to use this as a starting point anyway, and let me know (or better yet, create a
merge request) for suggested changes. Otherwise, sticking to this template and setup recommendations should minimise the
pain of setting up a new project from scratch, managing dependencies, tests and deploying.

Good article on setting up a similar
template: https://mitelman.engineering/blog/python-best-practice/automating-python-best-practices-for-a-new-project/

Author: Sam West sam.west@csiro.au Please contact me if you have issues, suggestions or general feedback!

## Quickstart

To get started quickly, install a few things. See Setup section below for instructions.

Required (see `How to install stuff` section below for details):

- [git](https://gitforwindows.org) (latest version)
- [pyenv](https://github.com/pyenv/pyenv?tab=readme-ov-file#installation) or [pyenv-win](https://github.com/pyenv-win/pyenv-win?tab=readme-ov-file#installation)
  eg `c:\users\<USER>\python39`. Don't add to Path!
- [poetry](https://python-poetry.org/docs/#installation)

Optional:

- Docker

Then grab the code, build and run it from a command prompt (bash, cmd, powershell, whatever) like this. Tested in
windows, should be similar in other OSes:

### Automatic installer:

There's now an automatic install script which clones this repo, updates pyproject.toml, re-inits the git tracking and
builds the poetry env for you. Use it like this:

First, download https://github.com/csiro-energy-systems/_PythonTemplate/blob/master/scripts/install-template.py in your
web browser (click 'Raw' button > Right Click > Save as).
Run the installer like this:

```shell
cd your-code-parent-dir # eg %USERPROFILE%\code\ or ~\code
python path\to\install-template.py --help # for usage.
python path\to\install-template.py your-new-project-dir-name
```

### Manual installation

```shell

# Go to wherever you keep your code (eg c:\code), and run these commands.
# This will create a dir called your/code/dir/0_pythontemplate/ containing the all the template project code
cd your\code\dir
git clone https://bitbucket.csiro.au/scm/energy/0_pythontemplate.git



# Now tell poetry which python install to use. On windows with normal python, it'll be like: C:\Users\wes148\Python\Python39\python.exe
# This project was set up for Python 3.9 (current and stable at time of writing), but should work fine with newer (and probably older) distros.
# It's best to use a vanilla distro (from https://www.python.org/downloads/) without additional packages installed, rather than an existing
# virtualenv or distro with additional included packages (like Anaconda), as we want Poetry to manage *all* your dependencies.
# It's fine to install multiple python distros, just make sure that `python --version` is the one you want to use.
poetry env use path/to/python3.9.exe

# Set credentials for private pypi package index - allows you to `poetry add <package>` any package from https://dev.azure.com/csiro-energy/csiro-energy/_artifacts/feed/csiro-python-packages
# Ask Sam West or Matt Amos for access.
# First, uncomment the `[[tool.poetry.source]]` section in pyproject.toml, then run:
poetry config repositories.csiroenergy https://pkgs.dev.azure.com/csiro-energy/csiro-energy/_packaging/csiro-python-packages/pypi/upload
# Use Nexus ident & password, or generate a new token at https://dev.azure.com/csiro-energy/_usersSettings/tokens.
# Linux users may get `keyring` errors from this command, see solution here: https://blog.frank-mich.com/python-poetry-1-0-0-private-repo-issue-fix/
poetry config http-basic.csiroenergy <ident> <password-or-token>

poetry install # install all dependencies (direct and transitive) listed in project.toml
poetry shell # activate virtual environment
python --version # make sure this is v3.9 (or whatever your set above with `poetry env use ...`).
pre-commit install # install pre-commit hooks for code quality checks. See Pre-commit section below.
python src\YOUR_MODULE_NAME\example.py #run code
pytest # run unit tests
docker-compose build #build a docker image, and define two containers, one for tests, and one for launching your code
docker-compose up #run docker containers
```

Then to start writing your own code, rename the source dir as appropriate:

```shell
cd YOUR_PROJECT_NAME
rmdir .git /S #remove the existing git repo dir, because you'll want to commit it as a new project
rename python_template YOUR_PROJECT_NAME # rename the source dir to match your project's top level package name.  Usually just the project name.
   # Also manually replace `python_template` with `YOUR_PROJECT_NAME` in `pyproject.toml`'s `name` and `packages` sections.
poetry add PACKAGE1 PACKAGE2 #install whatever packages you need, eg pandas, plotly, etc.
edit Readme.md # Write some doc of your own
edit docker-compose.yml # Change names to match your project name and launch script.
```

Then use your favourite git client to push it back to a new repo when ready, or from a shell:

```shell
git init #initialise local repo - recreates .git dir we deleted earlier
git add remote NEW_REPO_URL
notepad .gitignore # make sure you add non-source code files to .gitignore
git commit -a -m "commit message"
git push
```

## Setup

You'll need some stuff installed before starting, if you don't have them already.

### How to install stuff

- Git
    - Windows: https://git-scm.com/download/win
    - Linux: `apt-get install git`

- Python 3.9: https://www.python.org/downloads/
    - DO NOT select any option to add python to your PATH.
    - Poetry _can_ just use existing python (or conda) installations, but to avoid potential issues, it's much cleaner
      to just install vanilla python versions somewhere easy to remember.

- Poetry:
    - Full install instructions are here: https://python-poetry.org/docs/
    - DON'T USE `pip install poetry` or `conda install poetry`, as these only install it to the current env (which
      breaks things later), not as a system-wide standalone app.
    - OSX/Linux:
        - `curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -`
    - Windows (from PowerShell):
        - `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -`
        - Note: you might have to replace `| python` with `| c:\path\to\python.exe` if no python.exe was found on your
          PATH environment variable. for example:
            - `(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | c:\Users\wes148\Python\Python39\python.exe`

- Pycharm Community Edition: https://www.jetbrains.com/pycharm/download/
- Pycharm Poetry plugin: https://plugins.jetbrains.com/plugin/14307-poetry

- Docker: https://docs.docker.com/get-docker
    - Windows:
        - Upgrade to Windows version 1903 (Build 18362 or higher) or newer (`Start Menu > Run > "winver"` to get your
          version)
            - If you have an older version,
        - Install WSL2: https://docs.microsoft.com/en-us/windows/wsl/install-win10#manual-installation-steps.
        - This boils down to (run commands from elevated, aka Administrator, powershell):
            - Become an admin via `MakeMeAdmin` in the usual way
            - `Start Menu > type "powershell" > press Ctrl+Shift+Enter`
            - `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`
            - Install this: https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi
            - `wsl --set-default-version 2` (if this fails, don't worry just keep going)
            - You'll probably need to reboot at this point
            - (Possibly run this
              again :) `dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart`)
            - Install Ubuntu: https://www.microsoft.com/store/apps/9n6svws3rx71
            - Install `Docker Desktop on Windows`: https://docs.docker.com/get-docker
    - Linux:
        - Follow the instructions for your distro
          found [here on docker.com](https://docs.docker.com/engine/install/#server)
        - To install docker-compose
          follow [these instructions](https://docs.docker.com/compose/install/#install-compose)

## Managing Dependencies with Poetry

As mentioned in [Poetry's documentation](https://python-poetry.org/docs/basic-usage#specifying-dependencies), poetry
dependencies are stored in the tool.poetry.dependencies section of the pyproject.toml file.

The poetry CLI can manage adding a new dependency for you with the `add` command: `poetry add <dependency>`

## Continuous Integration

### Github Actions

Github provides built-in continuous integration via Github Actions. This will be run after any push to a github repo
with Actions enabled.
A basic poetry-based github action is provided in `.github/workflows/poetry-tests.yml` which:

- Builds the poetry environment in a Ubuntu container
- Runs pre-commit hooks on all python files in the repo via `pre-commit run --all-files`
- Runs unit tests via `pytest`

Committers will be emailed with results, and details can be accessed from the `Actions` tab in your github repo page.

### Installing from github:

Once your project is in github, you can then re-use it from other projects by just running something like:

``` poetry add git+ssh://git@github.com/csiro-energy-systems/YOUR_PROJECT_NAME.git#master ``` (if you've set up ssh keys
int github)

or

``` poetry add git+https://github.com/csiro-energy-systems/YOUR_PROJECT_NAME.git#master ``` (if you've already saved
your https github credentials)

If you must, you can also `pip install` from these same URLs.

### Jenkins (Deprecated!)

_The included Jenkinsfile should still work if you really want to host& maintain your own server. Github Actions is the
recommended replacement now though._

An example `Jenkinsfile` is provided. This defines a Jenkins continuous integration job pipeline which sets up this
project's Poetry environment inside a python docker container (set agent/docker/image below to your project's python
version), runs unit tests using pytest, and reports success/failure back to bitbucket (you get a new 'Build' column next
to commits) and via email to the last committer.

Jenkins is configured to scan https://bitbucket.csiro.au/projects/ENERGY periodically and build any changed branches
with a `Jenkinsfile` in their root, so you should get this functionality without having to do anything.

If you want to create a CI pipeline for a project outside of https://bitbucket.csiro.au/projects/ENERGY, you can log
into http://en-22-cdc.it.csiro.au:8080/ and do so manually (create a new `Multibranch Pipeline`).

You can also integrate Jenkins with mode IDEs also, eg:

* PyCharm: https://github.com/MCMicS/jenkins-control-plugin/issues/123#issuecomment-643127166
* VS Code: https://marketplace.visualstudio.com/items?itemName=ms-vsts.services-jenkins

## Integrated Development Environments (IDE)

This is a personal choice, but we recommend the following, as they generally make life easier, and your code better:

* PyCharm www.jetbrains.com/pycharm/download/ (Community version is free, Professional is paid)
* Visual Studio Code https://code.visualstudio.com/

## Git Clients

Commandline is fine for the basics. GUIs are generally easier, especially for resolving conflicts. Try:

* SmartGit https://www.syntevo.com/smartgit/ (paid, great merge editor)
* GitKraken https://www.gitkraken.com/ (paid but humorous name and pretty)
* Sourcetree https://www.sourcetreeapp.com/ (free once you register against bitbucket.csiro.au)
* Github Desktop: https://desktop.github.com (free)

## Code Style & Linters

The Python code style bible is [PEP8](https://peps.python.org/pep-0008/) and is well worth a quick read.
There are plugins (aka Linters) for most IDEs that warn you of style errors, plus add lots of useful code-time static
error checking.
It's a really good idea to install at least one of these:

PyCharm:

* Pylint: https://plugins.jetbrains.com/plugin/11084-pylint
* Sonarlint: https://plugins.jetbrains.com/plugin/7973-sonarlint

VS Code:

* Pylance: https://github.com/microsoft/pylance-release
* Flake8: https://stackoverflow.com/questions/54160207/using-flake8-in-vscode

A selection of commandline code quality checks are already part of this template's pre-commit hooks, and can be run on
staged changes via:
` poetry run pre-commit ` Or on all files via: ` poetry run pre-commit --all-files `

## Code Comments

Use reStructuredText formatting. See example code for details
Also see: https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html

## Git Ignores

You should only push code, scripts and config files to git. Basically only the stuff needed to build and test your
code.
Add wildcard patterns to `.gitignore` so git doesn't push rubbish. Don't commit data to a git repo, unless it's a small
sample used by a test. Large data should go on a shared drive and be referenced via a URL in a config file, for example.

## Config file(s)

You should export any settings your program needs to a separate config file, or files. This allows it to be easily
located and changed at runtime without rebuilding or modifying your actual code. This is particularly useful with a
built docker image as this config can be changed and applied without rebuilding the whole image.

A simple example is provided at `src/config/config.py`. It just contains the logging config in this template, but you
can expand it as necessary. The format doesn't really matter (though just using python is often a good idea), and other
common formats include yaml, ini, json, toml etc.

## Logging

All non-trivial programs and scripts should use a logger.

_*Please* don't use `print()` everywhere_, someone (probably you) will eventually have to go and find replace all your
print statements when you realise that there are too many of them, they're hard to trace, you don't know where they're
coming from, and they don't get saved anywhere!

The standard python logging library is ok, but requires a lot of duplicated boilerplate code, and doesn't work well with
multi-processing. loguru (https://github.com/Delgan/loguru) solves these issues, and is a lot easier to use. The
provided logging configuration will log to both the console, and a file simultaneously, making debugging stuff later
much easier.

To use logging, from your main() function(s), do this:

```python
from loguru import logger
from config import log_config  # load special python file we'e created containing runtime configuration.

logger.configure(**config.logging)
```

And from all other files, just get a child logger using:

```python
from loguru import logger

logger.trace("Your message")
logger.debug("Your message")
logger.info("Your message")
logger.warning("Your message")
logger.error("Your message")
```

## Unit Tests

To run all unit tests in /test, do this:

```shell
pytest
```

## Test Coverage

To check your test coverage (i.e. how much of your code is actually executed when running unit tests) do this:

```shell
coverage run --source ./src -m pytest
coverage report
```

## Software licensing

This is a big topic, but some good resources are:

* [CSIRO License List](https://confluence.csiro.au/display/OSS/Open+Source+Software+-+Review+of+Common+Licences) - has a
  good list of common software licenses and which ones to avoid.
* [TLDR Legal](https://tldrlegal.com/) - great for quickly checking a license not listed in the CSIRO page above.

You can list all the licenses being used in this project by running:

```shell
scripts/export-licenses
```

## Pre-commit Hooks

Several precommit hooks are included in `.pre-commit-config.yaml`, all of which have a configuration section
in `pyproject.toml`.
They perform code formatting, code quality and type-safety checks, among other things.
You should enable them by running `poetry run pre-commit install` after cloning the project, which will make them run on
changed files before every commmit.
You can also run them manually on all files via `poetry run pre-commit run --all-files`.

## Code Auto-Formatting

Autopep8 is use for automatic code formatting, followed by the flake8 linter for additional code quality warnings.
These are run as precommit hooks, and from Github actions..

## API Doc Generation

API doc can be written as reStructuredText or Markdown in the `docs/source` directory. It can then be built as HTML
using `scripts/gen-doc.[sh|cmd]`.

## Docker

Docker is a system-independent container system that hosts all of your code in a virtual machine image (kind of) that
can be easily redeployed to another host very simply and reliably.

A basic `Dockerfile` and `docker-compose.yml` are provided to get you up and running quickly.

- `Dockerfile` defines the entire runtime environment, and specifies how the VM is built, where your code go, and how to
  run it.
- `docker-compose.yml` specifies how to actually run the docker container.

Install docker from https://www.docker.com/products/docker-desktop, then run:

- `docker-compose build` to build the container. This will take a while the first time, but caches all the filesystem
  layers for speed next time.
- `docker image ls` to see your built images
- `docker-compose up` to run it.
- `docker-compose up -d` to run it detached (so it stays running after you close the shell).
- `docker-compose run --entrypoint /bin/bash your_project_name` to run a shell inside the container for debugging
- `docker save python_template:latest | gzip > your_project_name.docker.tar.gz` to save it to a file for distribution.
    - The resulting image file is about 400Mb

## Amazon Web Services/Terraform

A basic Terraform script for creating a Ubuntu virtual machine in Amazon Web Services (AWS) is provided
in `scripts/main.tf`. This will create a EC2 VM instance on an existing AWS account, reconfigure the SSH port, and output its IP address and
SSH key, ready for deploying code to.

See detailed instructions for its usage in the script's header comments.

## Contributing

Please add/change/correct stuff here if needed. Pushes to `master` branch are disabled, so please make a new branch (
e.g. `feature\my-new-branch`) and
submit a Pull Request with Sam West and Matt Amos as reviewers when ready.

## Troubleshooting

- If you get `ModuleNotFoundError`s when running `pytest` or scripts, make sure you aren't using alpha/beta poetry
  versions. Also check that `/src` appears in the PythonPath when
  running: `poetry run python -c "import sys; print(sys.path)"`
- If you get `Access Denied` errors on windows when running poetry commands, close all python processes (or kill them
  all with: `taskkill /f /im python.exe`) and try again
