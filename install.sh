
set -ex

SCRIPT_PATH=$(dirname "$0")

$SCRIPT_PATH/install_conda.sh
$SCRIPT_PATH/install_dependencies.sh
$SCRIPT_PATH/install_python_dependencies.sh
