import os
import sys
import importlib.util
from pathlib import Path
import json
import math
import shutil
import sysconfig
import subprocess
import pybind11

#################################################
npu_utils_so_name = "npu_utils.so"
npu_utils_src_name = "npu_utils.cpp"
npu_utils_mod_name = "npu_utils"
launcher_so_name = "launcher.so"
launcher_src_name = "launcher.cpp"
launcher_mod_name = "__triton_launcher"
kernel_binary_name = "kernel.o"
json_name = "kernel_info.json"
# FIMXE: how to assign different device on remote machine?
device = 0
#################################################

if len(sys.argv) < 2:
    build_extra_lib_flags = None
else:
    build_extra_lib_flags = sys.argv[1]

script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)


def get_ascend_path() -> str:
    """
    Retrieves the path to the Ascend toolkit installation directory.

    This function gets the Ascend toolkit path from the ASCEND_HOME_PATH environment variable.
    If the environment variable is not set, it raises an EnvironmentError with instructions
    to source the appropriate setup script.

    Returns:
    str: The path to the Ascend toolkit installation directory

    Raises:
    EnvironmentError: If the ASCEND_HOME_PATH environment variable is not set
    """
    path = os.getenv("ASCEND_HOME_PATH", "")
    if path == "":
        raise EnvironmentError(
            "ASCEND_HOME_PATH is not set, source <ascend-toolkit>/set_env.sh first"
        )
    return Path(path)


def build_so(so_name: str, src_path, extra_lib_flags) -> str:
    """
    Compiles a C++ source file into a shared object (.so) file with specific dependencies.

    This function constructs and executes a compiler command to build a shared library
    from a C++ source file, including necessary include paths and libraries for Python
    and Ascend platform dependencies. It automatically detects a suitable C++ compiler
    (clang++ or g++) if not specified through the CC environment variable.

    Args:
    so_name: Name of the output shared object file
    src_path: Path to the C++ source file to be compiled
    extra_lib_flags: Additional library flags to pass to the compiler

    Returns:
    Path to the generated shared object file

    Raises:
    RuntimeError: If no C++ compiler is found or if compilation fails
    """
    src_dir = os.path.dirname(src_path)
    so_path = os.path.join(src_dir, so_name)

    cxx = os.environ.get("CC")
    if cxx is None:
        clangxx = shutil.which("clang++")
        gxx = shutil.which("g++")
        cxx = clangxx if clangxx is not None else gxx
        if cxx is None:
            raise RuntimeError("Failed to find C++ compiler")
    cc_cmd = [cxx, src_path]
    # disable all warnings
    cc_cmd += [f"-w"]
    # find the python library
    if hasattr(sysconfig, "get_default_scheme"):
        scheme = sysconfig.get_default_scheme()
    else:
        scheme = sysconfig._get_default_scheme()
    # 'posix_local' is a custom scheme on Debian. However, starting Python 3.10, the default install
    # path changes to include 'local'. This change is required to use triton with system-wide python.
    if scheme == "posix_local":
        scheme = "posix_prefix"
    py_include_dir = sysconfig.get_paths(scheme=scheme)["include"]
    cc_cmd += [f"-I{py_include_dir}"]
    # find the ascend library
    asc_path = get_ascend_path()
    cc_cmd += [
        f"-I{os.path.join(asc_path, 'include')}",
        f"-I{os.path.join(asc_path, 'include/experiment')}",
        f"-I{os.path.join(asc_path, 'include/experiment/msprof')}",
        f"-I{pybind11.get_include()}",
    ]
    if extra_lib_flags:
        cc_cmd += [extra_lib_flags]
    cc_cmd += [
        f"-L{os.path.join(asc_path, 'lib64')}",
        "-lruntime",
        "-lascendcl",
    ]

    cc_cmd += ["-std=c++17", "-shared", "-fPIC", "-o", so_path]

    ret = subprocess.check_call(cc_cmd)

    if ret == 0:
        return so_path
    raise RuntimeError("Failed to compile " + src_path)


def readConfigFromJson(fpath: str):
    with open(fpath, 'r') as f:
        loaded_data = json.load(f)
    return loaded_data


def getModule(moduleName: str, so_path: str):
    spec = importlib.util.spec_from_file_location(moduleName, so_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def loadArg(packed_arg, bin_root_dir):
    filename = packed_arg["filename"]
    dtype_str = packed_arg["dtype"]
    shape = packed_arg["shape"]
    num_bytes_in_elem = mapDType2NumBytes(dtype_str)
    numel = math.prod(shape)
    num_bytes_all_numel = num_bytes_in_elem * numel
    tensor_hst = npu_utils_mod.allocate_host_memory(num_bytes_all_numel)
    bin_fpath = os.path.join(bin_root_dir, filename)
    npu_utils_mod.read_data_from_file(bin_fpath, tensor_hst)
    return tensor_hst


def saveData(packed_arg, bin_root_dir, tensor_hst):
    filename = packed_arg["filename"]
    dtype_str = packed_arg["dtype"]
    shape = packed_arg["shape"]
    num_bytes_in_elem = mapDType2NumBytes(dtype_str)
    numel = math.prod(shape)
    num_bytes_all_numel = num_bytes_in_elem * numel
    bin_fpath = os.path.join(bin_root_dir, filename)
    npu_utils_mod.write_data_to_file(bin_fpath, tensor_hst, num_bytes_all_numel)


def copyToDevice(packed_arg, tensor_hst):
    dtype_str = packed_arg["dtype"]
    shape = packed_arg["shape"]
    num_bytes_in_elem = mapDType2NumBytes(dtype_str)
    numel = math.prod(shape)
    num_bytes_all_numel = num_bytes_in_elem * numel
    tensor_dev = npu_utils_mod.allocate_device_memory(num_bytes_all_numel)
    npu_utils_mod.copy_memory(tensor_dev, tensor_hst, num_bytes_all_numel, "H2D")
    return tensor_dev


def copyFromDevice(packed_arg, tensor_hst, tensor_dev):
    dtype_str = packed_arg["dtype"]
    shape = packed_arg["shape"]
    num_bytes_in_elem = mapDType2NumBytes(dtype_str)
    numel = math.prod(shape)
    num_bytes_all_numel = num_bytes_in_elem * numel
    npu_utils_mod.copy_memory(tensor_hst, tensor_dev, num_bytes_all_numel, "D2H")


def mapDType2NumBytes(dtype_str):
    dtype_to_size = {
        'torch.float32': 4,
        'torch.float64': 8,
        'torch.float16': 2,
        'torch.bfloat16': 2,
        'torch.complex32': 4,
        'torch.complex64': 8,
        'torch.complex128': 16,
        'torch.int8': 1,
        'torch.uint8': 1,
        'torch.int16': 2,
        'torch.int32': 4,
        'torch.int64': 8,
        'torch.bool': 1,
    }
    dtype_str = dtype_str.strip("'\"")
    return dtype_to_size.get(dtype_str)
#################################################

json_path = os.path.join(script_dir, json_name)
configs = readConfigFromJson(json_path)

npu_utils_src_path = os.path.join(script_dir, npu_utils_src_name)
npu_utils_so_path = build_so(npu_utils_so_name, npu_utils_src_path, extra_lib_flags=build_extra_lib_flags)
npu_utils_mod = getModule(npu_utils_mod_name, npu_utils_so_path)

kernel_path = os.path.join(script_dir, kernel_binary_name)
kernel = Path(kernel_path).read_bytes()

fnname = configs["packed_metadata"]["kernel_name"]
mix_mode = configs["mix_mode"]
shared = configs["shared"]
mod, func, _, _ = npu_utils_mod.load_kernel_binary(fnname, kernel, shared, device, mix_mode)
launcher_so_path = build_so(launcher_so_name, launcher_src_name, extra_lib_flags=build_extra_lib_flags)
launcher_mod = getModule(launcher_mod_name, launcher_so_path)

stream = npu_utils_mod.create_stream()

gridX = configs["gridX"]
gridY = configs["gridY"]
gridZ = configs["gridZ"]
packed_metadata = configs["packed_metadata"]

num_arg = configs["num_arg"]
func_args = ()
hst_tensors = []
dev_tensors = []
for i in range(num_arg):
    packed_arg = configs[f"arg_{i}"]
    if (isinstance(packed_arg, dict)):
        tensor_hst = loadArg(packed_arg, script_dir)
        tensor_dev = copyToDevice(packed_arg, tensor_hst)
        arg = tensor_dev
        hst_tensors.append(tensor_hst)
        dev_tensors.append(tensor_dev)
    else:
        arg = packed_arg
        hst_tensors.append(0)
        dev_tensors.append(0)
    func_args += (arg,)

launch_metadata = None
launch_enter_hook = None
launch_exit_hook = None

args = (gridX, gridY, gridZ, stream, func, packed_metadata, launch_metadata, launch_enter_hook, launch_exit_hook)
args += func_args
kwargs = configs["kwargs"]

profiler_registered = launcher_mod.launch(*args, **kwargs)

for i in range(num_arg):
    packed_arg = configs[f"arg_{i}"]
    if (isinstance(packed_arg, dict)):
        copyFromDevice(packed_arg, hst_tensors[i], dev_tensors[i])
        saveData(packed_arg, script_dir, hst_tensors[i])
    else:
        pass