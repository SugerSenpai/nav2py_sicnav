#!/usr/bin/env python3
import os
import subprocess
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(command, cwd=None):
    """Run a shell command and handle errors."""
    try:
        result = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
        logger.info(f"Command succeeded: {' '.join(command)}")
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {' '.join(command)}\nError: {e.stderr}")
        raise

def main():
    # Paths
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "nav2py_sicnav_controller"))
    rvo2_dir = os.path.join(base_dir, "safe-interactive-crowdnav", "Python-RVO2")
    venv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "build", "nav2py_sicnav_controller", "venv"))

    # Check virtual environment
    if not os.path.exists(venv_dir):
        logger.error(f"Virtual environment not found at {venv_dir}. Run colcon build first.")
        sys.exit(1)

    # Check if Python-RVO2 exists
    if not os.path.exists(rvo2_dir):
        logger.warning(f"Python-RVO2 not found at {rvo2_dir}. Attempting to clone...")
        sicnav_dir = os.path.join(base_dir, "safe-interactive-crowdnav")
        try:
            run_command(["git", "clone", "https://github.com/sybrenstuvel/Python-RVO2.git"], cwd=sicnav_dir)
        except subprocess.CalledProcessError:
            logger.error("Failed to clone Python-RVO2. Ensure git is installed and accessible.")
            sys.exit(1)

    # Verify Python-RVO2 directory
    if not os.path.exists(rvo2_dir):
        logger.error(f"Python-RVO2 directory still missing at {rvo2_dir} after cloning attempt.")
        sys.exit(1)

    # Use virtual environment's pip
    pip_bin = os.path.join(venv_dir, "bin", "pip")

    # Verify Cython is installed
    try:
        run_command([pip_bin, "show", "cython"])
        logger.info("Cython is installed in virtual environment")
    except subprocess.CalledProcessError:
        logger.error("Cython is not installed. Ensure it is listed in pyproject.toml dependencies.")
        sys.exit(1)

    # Install rvo2
    try:
        logger.info(f"Installing rvo2 from {rvo2_dir}")
        run_command([pip_bin, "install", "-e", rvo2_dir])
        logger.info("Successfully installed rvo2")
    except subprocess.CalledProcessError:
        logger.error("Failed to install rvo2")
        sys.exit(1)

if __name__ == "__main__":
    main()