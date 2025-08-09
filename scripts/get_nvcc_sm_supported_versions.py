#!/usr/bin/env python3

"""
Script to identify which CUDA versions support which SM architectures.
The script downloads NVCC packages from NVIDIA repositories and extracts
the supported SM architectures from each version.
"""

import argparse
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Union

import requests

# Global constants
NVIDIA_CUDA_REPOS_URL = "https://developer.download.nvidia.com/compute/cuda/repos"
CUDA_ARCHIVE_URL = "https://developer.nvidia.com/cuda-toolkit-archive"


def sm_sort_key(sm_ver_str: str) -> tuple[int, str]:
    """Helper function to extract the numeric and alphabetical parts of an SM version string for sorting."""
    match = re.search(r"sm_(\d+)([af]?)", sm_ver_str)
    if match:
        numeric_part = int(match.group(1))
        alpha_part = match.group(2) or ""
        return (numeric_part, alpha_part)
    return (0, "")


class Version:
    """Class to handle CUDA version parsing, comparison, and formatting."""

    def __init__(self, version_str: str):
        """
        Initialize a Version object from a string.

        Args:
            version_str: Version string in format "X.Y", "X-Y", or just "X"

        Raises:
            ValueError: If the version string cannot be parsed as a valid version
        """
        # Normalize to dot format for internal storage
        self.version_str = version_str.replace("-", ".")
        parts = self.version_str.split(".")

        # Validate that at least one component is present and it's a digit
        if not parts or not all(part.isdigit() for part in parts):
            raise ValueError(
                f"Invalid version format: {version_str}. Expected components to be digits."
            )

        # Store components as integers
        self.components = [int(part) for part in parts]
        self.major = self.components[0]
        self.minor = self.components[1] if len(self.components) > 1 else 0

    @property
    def major_only(self) -> bool:
        """Return True if this version contains only a major component."""
        return len(self.components) == 1

    def __str__(self) -> str:
        """Return the version as a string in standard format."""
        return self.version_str

    def __repr__(self) -> str:
        """Return the version as a string for debugging."""
        return f"Version('{self.version_str}')"

    def __eq__(self, other: Union["Version", str]) -> bool:
        """Compare versions for equality."""
        if isinstance(other, str):
            other = Version(other)

        # Compare all available components
        for i in range(max(len(self.components), len(other.components))):
            self_comp = self.components[i] if i < len(self.components) else 0
            other_comp = other.components[i] if i < len(other.components) else 0

            if self_comp != other_comp:
                return False

        return True

    def __lt__(self, other: Union["Version", str]) -> bool:
        """Compare versions for less than."""
        if isinstance(other, str):
            other = Version(other)

        # Compare all available components
        for i in range(max(len(self.components), len(other.components))):
            self_comp = self.components[i] if i < len(self.components) else 0
            other_comp = other.components[i] if i < len(other.components) else 0

            if self_comp < other_comp:
                return True
            elif self_comp > other_comp:
                return False

        return False  # Equal versions

    def __gt__(self, other: Union["Version", str]) -> bool:
        """Compare versions for greater than."""
        if isinstance(other, str):
            other = Version(other)

        # Compare all available components
        for i in range(max(len(self.components), len(other.components))):
            self_comp = self.components[i] if i < len(self.components) else 0
            other_comp = other.components[i] if i < len(other.components) else 0

            if self_comp > other_comp:
                return True
            elif self_comp < other_comp:
                return False

        return False  # Equal versions

    def __le__(self, other: Union["Version", str]) -> bool:
        """Compare versions for less than or equal."""
        return self < other or self == other

    def __ge__(self, other: Union["Version", str]) -> bool:
        """Compare versions for greater than or equal."""
        return self > other or self == other

    def __hash__(self) -> int:
        """Make Version objects hashable for use in sets/dicts."""
        return hash(tuple(self.components))

    def within_range(
        self, min_ver: Union["Version", str, None], max_ver: Union["Version", str, None]
    ) -> bool:
        """
        Check if this version is within the specified min/max range.

        Args:
            min_ver: Minimum version (inclusive) or None for no lower bound
            max_ver: Maximum version (inclusive) or None for no upper bound

        Returns:
            True if version is within range, False otherwise
        """
        if isinstance(min_ver, str):
            min_ver = Version(min_ver)

        if isinstance(max_ver, str):
            max_ver = Version(max_ver)

        return (min_ver is None or self >= min_ver) and (max_ver is None or self <= max_ver)

    def package_format(self) -> str:
        """Return version string in package format (X-Y)."""
        # For package format, we need at least major.minor
        if self.major_only:
            return f"{self.major}-0"
        return f"{self.major}-{self.minor}"


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Find SM architectures supported by different CUDA versions."
    )

    # Positional argument for specific CUDA versions
    parser.add_argument(
        "cuda_versions",
        nargs="*",
        help="Specific CUDA versions to check (e.g., '11.4' or '11' for all 11.x). "
        "If not provided, all available versions will be checked.",
    )

    # Optional arguments for version ranges
    parser.add_argument(
        "--min",
        type=str,
        help="Minimum CUDA version to check (e.g., '11.0'). Ignored if specific versions are provided.",
    )

    parser.add_argument(
        "--max",
        type=str,
        help="Maximum CUDA version to check (e.g., '12.2'). Ignored if specific versions are provided.",
    )

    # Table formatting options
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Generate a compact table without extra spacing",
    )

    # SM filtering options
    parser.add_argument(
        "-s",
        "--list-specifics",
        action="store_true",
        help="List arch-specific or family-specific SM versions (e.g., sm_90a, sm_120f) in addition to base versions.",
    )

    return parser.parse_args()


def setup_temp_dir() -> Path:
    """Create a temporary directory for downloads and extractions."""
    temp_dir = Path(tempfile.mkdtemp())
    print(f"Temp directory: {temp_dir}")
    return temp_dir


def get_ubuntu_distros() -> list[str]:
    """Get list of Ubuntu distributions from NVIDIA website."""
    print(f"Fetching available repositories from {NVIDIA_CUDA_REPOS_URL}...")

    try:
        response = requests.get(f"{NVIDIA_CUDA_REPOS_URL}/")
        if response.status_code != 200:
            print(f"ERROR: Failed to fetch repositories, status code: {response.status_code}")
            return []

        # Find all Ubuntu distributions in the response
        ubuntu_distros = sorted(
            set(re.findall(r"ubuntu[0-9]+", response.text)),
            key=lambda x: [int(n) for n in re.findall(r"\d+", x)],
            reverse=True,
        )

        if not ubuntu_distros:
            print("ERROR: Could not find any Ubuntu distributions in the response")
            return []

        print(f"Found {len(ubuntu_distros)} Ubuntu distributions: {' '.join(ubuntu_distros)}")
        return ubuntu_distros

    except Exception as e:
        print(f"ERROR: Failed to fetch Ubuntu distributions: {str(e)}")
        return []


def get_cuda_versions() -> list[Version]:
    """Get list of CUDA toolkit versions from NVIDIA website."""
    print(f"Fetching available CUDA toolkit versions from {CUDA_ARCHIVE_URL}...")

    try:
        response = requests.get(CUDA_ARCHIVE_URL)
        if response.status_code != 200:
            print(
                f"ERROR: Failed to fetch CUDA toolkit archive, status code: {response.status_code}"
            )
            return []

        # Find all "Toolkit X.Y" occurrences and extract X.Y
        toolkit_pattern = r"Toolkit (\d+)\.(\d+)"
        matches = re.findall(toolkit_pattern, response.text)

        if not matches:
            print("ERROR: Could not find any CUDA toolkit versions in the response")
            return []

        # Create and sort Version objects (newest first)
        cuda_versions = sorted(
            {Version(f"{major}.{minor}") for major, minor in matches}, reverse=True
        )
        return cuda_versions

    except Exception as e:
        print(f"ERROR: Failed to fetch CUDA toolkit versions: {str(e)}")
        return []


def filter_cuda_versions(
    all_versions: list[Version],
    requested_versions: list[str] = None,
    min_version: str = None,
    max_version: str = None,
) -> list[Version]:
    """
    Filter CUDA versions based on user specifications.

    Args:
        all_versions: list of all available CUDA versions
        requested_versions: list of specific versions requested by user
        min_version: Minimum CUDA version to include
        max_version: Maximum CUDA version to include

    Returns:
        Filtered list of CUDA versions to check
    """
    # Parse min/max version bounds if provided
    min_ver = None
    if min_version:
        try:
            min_ver = Version(min_version)
        except ValueError:
            print(f"WARNING: Invalid minimum version format: {min_version}. Ignoring min filter.")

    max_ver = None
    if max_version:
        try:
            max_ver = Version(max_version)
        except ValueError:
            print(f"WARNING: Invalid maximum version format: {max_version}. Ignoring max filter.")

    # If no specific versions are requested, just apply min/max filters to all
    if not requested_versions:
        return [v for v in all_versions if v.within_range(min_ver, max_ver)]

    # Process specific version requests
    filtered_versions = []
    for version_str in requested_versions:
        try:
            # Try to parse as a version (can be major-only or major.minor)
            version = Version(version_str)

            if version.major_only:
                # If it's a major-only version like "11", match all with same major
                major = version.major
                major_versions = [
                    v for v in all_versions if v.major == major and v.within_range(min_ver, max_ver)
                ]
                filtered_versions.extend(major_versions)
            else:
                # It's a specific version like "11.4"
                if version in all_versions and version.within_range(min_ver, max_ver):
                    filtered_versions.append(version)
        except ValueError:
            print(f"WARNING: Invalid version format: {version_str}. Skipping.")

    # Remove duplicates and sort
    return sorted(set(filtered_versions), reverse=True)


def process_cuda_version(
    cuda_version: Version, distros: list[str], temp_dir: Path, list_specifics: bool
) -> tuple[bool, set[str]]:
    """
    Process a CUDA version to find supported SM architectures.

    Args:
        cuda_version: CUDA toolkit version
        distros: list of Ubuntu distributions to try
        temp_dir: Temporary directory for downloads

    Returns:
        tuple containing:
            - Boolean indicating success
            - set of SM architecture strings supported by this CUDA version
    """

    for distro in distros:
        repo_url = f"{NVIDIA_CUDA_REPOS_URL}/{distro}/x86_64"

        # Check if repository exists
        try:
            reponse_head = requests.head(f"{repo_url}/")
            if reponse_head.status_code != 200:
                continue
        except Exception:
            continue

        # Get list of packages in the repository and process
        try:
            response_get = requests.get(f"{repo_url}/")
            if response_get.status_code != 200:
                continue

            # Find NVCC package for this CUDA version - convert to package format
            packages = re.findall(
                rf"cuda-nvcc-{cuda_version.package_format()}_[0-9\.-]*_amd64\.deb",
                response_get.text,
            )
            if not packages:
                print(
                    f"CUDA {cuda_version}: Could not find package for {cuda_version} in the {distro} CUDA repository"
                )
                continue

            # Sort and take the latest package
            packages.sort(
                key=lambda x: [int(n) if n.isdigit() else n for n in re.findall(r"\d+|\D+", x)]
            )
            latest_package = packages[-1]

            print(f"CUDA {cuda_version}: Found {latest_package} in the {distro} CUDA repository")

            # Download the package
            download_url = f"{repo_url}/{latest_package}"
            download_path = temp_dir / latest_package

            response_dl = requests.get(download_url, stream=True)
            if response_dl.status_code != 200:
                print(f"Failed to download {download_url}")
                continue

            with open(download_path, "wb") as f:
                shutil.copyfileobj(response_dl.raw, f)

            # Extract the package
            extract_dir = temp_dir / f"extract_{cuda_version}"
            extract_dir.mkdir(exist_ok=True)

            subprocess.run(
                ["dpkg", "-x", str(download_path), str(extract_dir)],
                capture_output=True,
            )

            # Find nvcc binary
            nvcc_paths = []
            for root, _, files in os.walk(extract_dir):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    if file == "nvcc" and file_path.is_file() and os.access(file_path, os.X_OK):
                        nvcc_paths.append(file_path)

            if not nvcc_paths:
                print("Could not find nvcc in extracted package")
                continue

            nvcc_path = nvcc_paths[0]

            # Try to get supported SM versions from nvcc
            sm_versions = set()

            # Define the regex pattern based on whether to include specific versions
            sm_pattern = r"(sm_\d+)"
            if list_specifics:
                sm_pattern = r"(sm_\d+[af]?)"

            # TODO: We're ignoring this because it does not list arch and family specific versions
            # # First try --list-gpu-code
            # try:
            #     result_list_gpu = subprocess.run(
            #         [str(nvcc_path), "--list-gpu-code"],
            #         capture_output=True,
            #         text=True,
            #     )
            #     if result_list_gpu.returncode == 0:
            #         for line in result_list_gpu.stdout.splitlines():
            #             sm_match = re.search(sm_pattern, line)
            #             if sm_match:
            #                 sm_versions.add(sm_match.group(1))
            # except Exception:
            #     pass

            # If that didn't work, try to extract from help output
            if not sm_versions:
                try:
                    # Make sure to not consider values that were dropped but listed in --help by
                    # filtering that out with sed
                    help_cmd = (
                        str(nvcc_path) + " --help | sed -n '/Allowed values/,/^$/p'"
                    )
                    result_help = subprocess.run(
                        help_cmd,
                        capture_output=True,
                        text=True,
                        shell=True,
                    )
                    for line in result_help.stdout.splitlines() + result_help.stderr.splitlines():
                        for sm_match in re.finditer(sm_pattern, line):
                            sm_versions.add(sm_match.group(1))
                except Exception:
                    pass

            if not sm_versions:
                print(f"Could not determine SM versions for CUDA {cuda_version}")
                break

            # Print and store the supported SM versions
            sorted_sm_list = sorted(list(sm_versions), key=sm_sort_key)
            print(f"CUDA {cuda_version}: Supports {' '.join(sorted_sm_list)}")
            return True, sm_versions

        except Exception as e:
            print(f"Error processing package from {distro} for CUDA {cuda_version}: {str(e)}")
            continue

    return False, set()


def get_sm_compatibility(
    cuda_versions: list[Version], ubuntu_distros: list[str], temp_dir: Path, list_specifics: bool
) -> dict[str, list[Version]]:
    """
    Process CUDA versions to find SM compatibility.

    Args:
        cuda_versions: list of CUDA versions
        ubuntu_distros: list of Ubuntu distributions to try
        temp_dir: Temporary directory for downloads

    Returns:
        dict mapping SM versions to list of compatible CUDA versions
    """
    sm_version_map = {}

    for cuda_version in cuda_versions:
        success, found_sm_set = process_cuda_version(cuda_version, ubuntu_distros, temp_dir, list_specifics)

        if success:
            for sm_arch_str in found_sm_set:
                if sm_arch_str in sm_version_map:
                    sm_version_map[sm_arch_str].append(cuda_version)
                else:
                    sm_version_map[sm_arch_str] = [cuda_version]
        else:
            print(f"CUDA {cuda_version}: Could not find or process NVCC package")

    return sm_version_map


def generate_markdown_table(sm_version_map: dict[str, list[Version]], compact: bool = False) -> str:
    """
    Generate Markdown table from SM compatibility data.

    Args:
        sm_version_map: dict mapping SM versions to supported CUDA versions
        compact: Whether to generate a compact table without extra spacing
    """
    # Sort SM versions by numeric value - NOW USES GLOBAL HELPER
    sorted_sm_versions = sorted(
        sm_version_map.keys(),
        key=sm_sort_key,  # USE GLOBAL HELPER
    )

    # Get all unique CUDA versions
    all_cuda_versions = set()
    for versions in sm_version_map.values():
        all_cuda_versions.update(versions)

    # Sort CUDA versions
    sorted_cuda_versions = sorted(all_cuda_versions, reverse=True)

    # Build the table
    table = ["## CUDA and SM Architecture Compatibility Matrix", ""]

    # Header row with SM versions
    # Use a more descriptive header that indicates rows vs columns
    header_label = "CUDA Ver \\ SM Arch"
    header = f"| {header_label} |"
    for sm_ver in sorted_sm_versions:
        header += f" {sm_ver} |"
    table.append(header)

    # Calculate column widths based on headers
    header_width = len(header_label)

    # Separator row
    separator = "|---|" if compact else f"| {'-' * header_width} |"
    for sm_ver in sorted_sm_versions:
        separator += "---|" if compact else f"-{'-' * len(sm_ver)}-|"
    table.append(separator)

    # Data rows - calculate padding dynamically based on header width
    for cuda_ver in sorted_cuda_versions:
        version_str = str(cuda_ver)

        if not compact:
            # Pad to match header width
            padding = header_width - len(version_str)
            version_str = version_str + " " * padding

        row = f"| {version_str} |"

        for sm_ver in sorted_sm_versions:
            # Check if this CUDA version supports this SM version
            value = "X" if cuda_ver in sm_version_map[sm_ver] else ""
            if not compact:
                padding = len(sm_ver) - len(value)
                left_padding = padding // 2
                right_padding = padding - left_padding
                value = " " * left_padding + value + " " * right_padding

            row += f" {value} |"

        table.append(row)

    table.append("")
    table.append(f"Table generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return "\n".join(table)


def main():
    """Main function to coordinate the script execution."""
    # Parse command line arguments
    args = parse_args()

    # Create temporary directory for downloads
    temp_dir = setup_temp_dir()

    # Register cleanup to ensure temp dir is always removed
    atexit.register(lambda p=temp_dir: shutil.rmtree(str(p), ignore_errors=True))

    try:
        # Get Ubuntu distributions
        ubuntu_distros = get_ubuntu_distros()
        if not ubuntu_distros:
            sys.exit(1)

        # Determine which CUDA versions to process
        cuda_versions = []

        # Try to use explicit versions if provided
        if args.cuda_versions:
            try:
                # Try to create Version objects for all specified versions
                cuda_versions = [Version(v) for v in args.cuda_versions]
                print(f"Using explicit CUDA versions: {' '.join(args.cuda_versions)}")
            except ValueError:
                # If any version is invalid (not in X.Y format), we'll need to fetch all versions
                print("Some version specifications need pattern matching, fetching all versions...")
                cuda_versions = []  # Clear any partially parsed versions

        # If we couldn't use explicit versions, fetch all and apply filtering
        if not cuda_versions:
            all_available_versions = get_cuda_versions()
            versions_str = " ".join(str(v) for v in all_available_versions)
            print(f"Found {len(all_available_versions)} CUDA toolkit versions: {versions_str}")
            if not all_available_versions:
                print("Failed to fetch available CUDA versions")
                sys.exit(1)

            # Apply filtering based on any specified patterns and min/max bounds
            cuda_versions = filter_cuda_versions(
                all_available_versions, args.cuda_versions, args.min, args.max
            )

        if not cuda_versions:
            print("No CUDA versions to process after filtering")
            sys.exit(1)

        print(f"Processing CUDA versions: {' '.join(str(v) for v in cuda_versions)}")

        # Process CUDA versions to find SM compatibility
        sm_version_map = get_sm_compatibility(cuda_versions, ubuntu_distros, temp_dir, args.list_specifics)

        # Generate and print Markdown table
        table = generate_markdown_table(sm_version_map, args.compact)
        print(table)

    finally:
        # Since we registered with atexit, this is belt-and-suspenders,
        # ensuring cleanup happens immediately when possible
        shutil.rmtree(str(temp_dir), ignore_errors=True)
        # De-register the atexit handler
        atexit.unregister(lambda p=temp_dir: shutil.rmtree(str(p), ignore_errors=True))


if __name__ == "__main__":
    main()
