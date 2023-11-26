#!/bin/bash

set -e
CONDA="conda"

conda_add_channel() {
  channel_name="$1"
  if $CONDA config --show channels | grep -q ${channel_name}
  then
      echo "Conda channel ${channel_name} is known already"
  else
      $CONDA config --add channels ${channel_name}
  fi
}

conda_rm_env() {
  ENV_NAME="${1}"
  if $CONDA env list | grep -q ${ENV_NAME}
  then
    echo "Remove existing $CONDA env ${ENV_NAME}"
    $CONDA env remove -n "${ENV_NAME}" -y
  fi
}

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

source "${DIR}/init_conda.sh"
set -u  # init $CONDA has some unbounded variables

ENV_NAME="revllm"
CONDA_PACKAGE_FILES="--file conda_packages.txt --file conda_packages_dev.txt"

conda_rm_env ${ENV_NAME}
conda_add_channel conda-forge
conda_add_channel pytorch

$CONDA create -n "${ENV_NAME}" --yes ${CONDA_PACKAGE_FILES}
$CONDA activate "${ENV_NAME}"

if [[ -f ".evn" ]]
then
    $CONDA env config vars set "$(paste -d ' ' -s "${DIR}/.env" | sed -e 's/"//g')"
fi

pip install --upgrade -r pip_packages.txt
$CONDA env list
