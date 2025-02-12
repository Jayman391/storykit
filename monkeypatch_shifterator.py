import os
import sys
import subprocess
import importlib.util
import importlib

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# --- Ensure shifterator is installed ---
spec = importlib.util.find_spec("shifterator")
if spec is None:
    install("shifterator")
    spec = importlib.util.find_spec("shifterator")
if spec is None:
    raise ImportError("shifterator installation failed")

# Get the package directory using the module’s search locations.
# (Don’t import shifterator yet so we can patch the source files first.)
shifterator_path = spec.submodule_search_locations[0]

# --- Patch plotting.py ---
plotting_path = os.path.join(shifterator_path, 'plotting.py')
with open(plotting_path, 'r') as f:
    lines = f.readlines()

# (Instead of relying solely on fixed line numbers, search for the target strings.)
# For example, change '"tight": True,' to '"tight": False,'
for i, line in enumerate(lines):
    if '"tight": True,' in line:
        lines[i] = line.replace('"tight": True,', '"tight": False,')
        break
else:
    print("Warning: Did not find '\"tight\": True,' in plotting.py")

# If you still want to change specific lines for the tick_params calls,
# you can check that the file is long enough and then modify the desired indices.
lines[744] = "    in_ax.tick_params(axis='y', labelsize=11)\n"
del lines[745]
lines[796] = "    in_ax.tick_params(axis='y', labelsize=12)\n"
del lines[797]
del lines[797]
del lines[797]

with open(plotting_path, 'w') as f:
    f.writelines(lines)

# --- Patch helper.py ---
helper_path = os.path.join(shifterator_path, 'helper.py')
with open(helper_path, 'r') as f:
    helper_lines = f.readlines()

# Replace 'collections.Mapping' with 'collections.abc.Mapping'
for i, line in enumerate(helper_lines):
    if "isinstance(scores, collections.Mapping)" in line:
        helper_lines[i] = line.replace("collections.Mapping", "collections.abc.Mapping")
        break
else:
    print("Warning: Did not find 'collections.Mapping' in helper.py")

with open(helper_path, 'w') as f:
    f.writelines(helper_lines)