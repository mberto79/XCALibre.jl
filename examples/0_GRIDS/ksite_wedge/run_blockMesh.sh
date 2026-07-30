#!/usr/bin/env bash
# =============================================================================
#  Run blockMesh for the K-Site wedge from a path that contains spaces.
# =============================================================================
#
#  OpenFOAM's `fileName` type does not permit spaces. When the repository lives
#  under something like
#
#      /mnt/c/Users/<you>/OneDrive - The University of Nottingham/Software Repo/...
#
#  blockMesh fails with:
#
#      fileName::stripInvalid() called for invalid fileName <path-with-spaces-removed>
#
#  because it silently strips the spaces and then cannot find the case. This
#  script sidesteps that by staging the case in a temporary space-free
#  directory, meshing there, and copying constant/polyMesh back.
#
#  Usage (from anywhere):
#      ./run_blockMesh.sh
#
#  Requires blockMesh on PATH (source your OpenFOAM bashrc first).
# =============================================================================

set -euo pipefail

# Directory containing this script (quoted throughout - spaces are fine for
# bash, it is only OpenFOAM that objects).
CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v blockMesh >/dev/null 2>&1; then
    echo "ERROR: blockMesh not found on PATH." >&2
    echo "       Source your OpenFOAM environment first, e.g.:" >&2
    echo "         source /opt/openfoam*/etc/bashrc" >&2
    echo "       or (openfoam.org builds):" >&2
    echo "         source /opt/openfoam*/etc/bashrc" >&2
    exit 1
fi

if [ ! -f "${CASE_DIR}/system/blockMeshDict" ]; then
    echo "ERROR: system/blockMeshDict not found." >&2
    echo "       Generate it first:  julia make_ksite_wedge.jl" >&2
    exit 1
fi

WORK_DIR="$(mktemp -d /tmp/ksite_wedge.XXXXXX)"
cleanup() { rm -rf "${WORK_DIR}"; }
trap cleanup EXIT

echo "Case      : ${CASE_DIR}"
echo "Staging in: ${WORK_DIR}"
echo

mkdir -p "${WORK_DIR}/system"
cp "${CASE_DIR}/system/blockMeshDict" "${WORK_DIR}/system/"
cp "${CASE_DIR}/system/controlDict"   "${WORK_DIR}/system/"

cd "${WORK_DIR}"
blockMesh

if [ ! -d "${WORK_DIR}/constant/polyMesh" ]; then
    echo >&2
    echo "ERROR: blockMesh finished but constant/polyMesh was not created." >&2
    exit 1
fi

# Copy the mesh back next to the dict, where FOAM3D_mesh expects it.
mkdir -p "${CASE_DIR}/constant"
rm -rf "${CASE_DIR}/constant/polyMesh"
cp -r "${WORK_DIR}/constant/polyMesh" "${CASE_DIR}/constant/"

echo
echo "Mesh written to: ${CASE_DIR}/constant/polyMesh"
echo
echo "Validate it from the repository root with:"
echo "    julia --project=. test/unit_test_ksite_wedge_mesh.jl"
