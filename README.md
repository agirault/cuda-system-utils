# CUDA System Utility Scripts

The [scripts](scripts) directory contains a collection of system utilities for working with the NVIDIA CUDA environment and toolchain. These scripts help in determining supported CUDA architectures, inspecting compiled CUDA binaries, managing CMake configurations.

---

### [`check_sm_support.sh`](./scripts/check_sm_support.sh)

#### Overview

This Bash script inspects compiled binaries (executables, shared libraries `.so`, static libraries `.a`) to determine the CUDA SM architectures they were compiled for. It reports both SASS (ELF cubin) and PTX (intermediate representation) architectures embedded in the files. The script uses the `cuobjdump` utility from the CUDA Toolkit to extract this information. It can process a single specified file or scan an entire directory for relevant files.

#### Prerequisites

*   Bash
*   CUDA Toolkit: The `cuobjdump` and `nvdisasm` command-line utilities must be installed and accessible either via the system's PATH or in the default location `/usr/local/cuda/bin/`.

#### Usage

```bash
./scripts/check_sm_support.sh <directory_or_file>
```

**Arguments**:

*   `<directory_or_file>`: The path to either a single file to inspect or a directory to scan. When a directory is provided, the script will search for executables, `.so` files, and `.a` files within it.

#### Output

For each file containing CUDA code, the script prints:

*   The file path.
*   A list of SASS (ELF) SM architectures found, along with a count if an architecture appears multiple times (e.g., `[1] sm_75`).
*   A list of PTX SM architectures found, marked with `(PTX)` (e.g., `[1] sm_70 (PTX)`).

#### Example

To check a specific compiled application:

```bash
./scripts/check_sm_support.sh /opt/my_app/bin/my_cuda_program
```

To scan all relevant files within a library directory:

```bash
./scripts/check_sm_support.sh /usr/local/nvidia/lib
```

---

### [`get_cmake_cuda_archs.py`](./scripts/get_cmake_cuda_archs.py)

#### Overview

This Python script determines and formats CUDA architectures suitable for use with CMake's `CUDA_ARCHITECTURES` variable. It takes a user request (e.g., 'all', 'all-major', 'native', or a specific list of architectures like '75 80 86a') and filters them based on the capabilities of the `nvcc` compiler found on the system. It can also filter by a minimum specified architecture (e.g., SM 7.0 and newer) and automatically excludes architectures corresponding to iGPUs when building on x86_64 platforms. It configures a PTX-only build (`virtual`) for the highest non-specific architecture, for forward compatibility, and optimized SASS builds (`real`) for all the architectures remaining.

The output is a semicolon-separated list formatted with `-real` and `-virtual` suffixes (e.g., `75-real;86-real;86-virtual`).

#### Prerequisites


*   Python 3
*   `nvcc` (NVIDIA CUDA Compiler): Must be accessible in the system's PATH or located at `/usr/local/cuda/bin/nvcc`. The script also allows specifying the `nvcc` path directly.

#### Usage

```bash
scripts/get_cmake_cuda_archs.py <requested_archs> [options]
```

**Arguments**:

*   `requested_archs`: Defines the architectures to consider.
    *   `'all'`: Use all architectures supported by the found `nvcc`.
    *   `'all-major'`: Use all major architectures (e.g., 70, 80, not 75, 86a) supported by `nvcc`.
    *   `'native'`: Prints the string "native" and exits (for CMake's `CUDA_ARCHITECTURES=native`).
    *   Comma or space-separated list of SM numbers (e.g., `'75 86 90a'`).

**Options**:

*   `--nvcc-path <path>`, `-n <path>`: Specify the full path to the `nvcc` executable.
*   `--min-arch <int>`, `-m <int>`: Set a minimum major CUDA architecture (e.g., `70` for Volta and newer). Architectures below this will be excluded.
*   `--verbose`, `-v`: Enable verbose debug logging to stderr.

#### Example

To get CMake formatted architectures for SM 7.5, 8.6, and 9.0a:

```bash
scripts/get_cmake_cuda_archs.py "75 86 90a"
```

To get all supported major architectures above 7.0:

```bash
scripts/get_cmake_cuda_archs.py all-major --min-arch 70
```

---

### [`get_nvcc_sm_supported_versions.py`](./scripts/get_nvcc_sm_supported_versions.py)

#### Overview

This Python script identifies which CUDA Toolkit versions support specific SM (Streaming Multiprocessor) architectures. It fetches data by downloading `cuda-nvcc` Debian packages from NVIDIA's official repositories for various Ubuntu distributions and CUDA toolkit versions. It then extracts the `nvcc` binary from these packages and runs it with `--help` to determine the supported SM architectures for that particular CUDA version.

The script aggregates this information and outputs a Markdown table mapping each SM architecture to the CUDA Toolkit versions that support it. By default, it lists base architectures (e.g., `sm_75`, `sm_80`), but can be configured to show arch-specific or family-specific versions as well (e.g. `sm_90a`, `sm_120f`).

#### Prerequisites

*   Python 3
*   `requests` Python package: Install via `pip install requests`.
*   `dpkg` command-line utility: Typically available on Debian-based systems (like Ubuntu). This is required to extract `.deb` packages.
*   Internet access: To download CUDA packages from NVIDIA and fetch lists of available versions.

#### Usage

```bash
scripts/get_nvcc_sm_supported_versions.py [cuda_versions...] [options]
```

**Arguments**:

*   `cuda_versions...`: (Optional) A list of specific CUDA versions to check (e.g., `11.4`, `12.0`). You can also specify just a major version (e.g., `11`) to check all minor versions within that major release (e.g., 11.0, 11.1, ...). If no versions are provided, the script attempts to check all discoverable CUDA versions.

**Options**:

*   `--min <version>`: Specify the minimum CUDA version to check (e.g., `11.0`). This is ignored if `cuda_versions` are explicitly provided.
*   `--max <version>`: Specify the maximum CUDA version to check (e.g., `12.2`). This is ignored if `cuda_versions` are explicitly provided.
*   `--compact`: Generate a more compact Markdown table without extra spacing between columns.
*   `-s`, `--list-specifics`: List arch-specific or family-specific SM versions (e.g., `sm_90a`) in addition to base versions. By default, only base versions are shown.

#### Example

To check which SM architectures are supported by CUDA 11.8, 12.0, and 12.1:

```bash
scripts/get_nvcc_sm_supported_versions.py 11.8 12.0 12.1
```

To check all CUDA versions from 11.0 up to (and including) 11.8:

```bash
scripts/get_nvcc_sm_supported_versions.py --min 11.0 --max 11.8
```

To check all available CUDA versions and output a compact summary Markdown table:

```bash
scripts/get_nvcc_sm_supported_versions.py --compact
```

To include arch-specific or family-specific SMs like `sm_90a` in the output:

```bash
scripts/get_nvcc_sm_supported_versions.py --list-specifics
```
