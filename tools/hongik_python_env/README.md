# Hongik Python environment (KIRO host)

Adapted from src/0001-Python-venv-from-a-YAML-spec-or-from-the-packages-se.patch.
Uses the delivery MANIFEST versions and links the delivered pure Python package source.
The system profile links apt NumPy/SciPy/PyYAML and their metadata inside the venv,
so existing user-site and /usr/local NumPy 2 cannot shadow them. No global packages
are removed, no complete system path is prepended to PYTHONPATH, and no pip/network
installation is needed for this profile.

Create: `/usr/bin/python3 tools/hongik_python_env/make_venv.py --profile system --dest ~/venv_hongik`
Activate: `source ~/venv_hongik/bin/activate`
Source ROS/workspace setup.bash for ROS module imports. The report is
`~/venv_hongik/hongik_venv_report.txt`.

ROS installed scripts with /usr/bin/python3 shebangs and explicit launch prefixes
still use system Python. Activation does not switch them to the venv.
