#!/usr/bin/env bash
# Usage: bash bootstrap-online.sh R DYNAMO_SRC ORIGINAL_EFFECTIVE_JSON
# New ARM workspace only. Main launches; this script does not launch serving/tests.
set -euo pipefail
R=${1:?fresh ARM workspace prefix}; SRC=${2:?staged Dynamo checkout}; PINS=${3:?original-effective-packages.json}
IMAGE_PY=/opt/sglang/bin/python3
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
[[ $(uname -m) == aarch64 && "$R" == /workspace/* && ! -e "$R/env" ]]
[[ $(git -C "$SRC" rev-parse HEAD) == f28d277edaf403ba5cae65bece687f7b29c0eba3 ]]
for tool in curl git gcc g++ pkg-config protoc; do command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 1; }; done
pkg-config --exists openssl hwloc libudev
mkdir -p "$R"/{build/wheels,sources,tmp,cache/pip,cargo,rustup}
trap 'rc=$?; printf "%s\n" "$rc" > "$R/bootstrap-online.exit"' EXIT
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV RUSTFLAGS CARGO_ENCODED_RUSTFLAGS CARGO_BUILD_TARGET
export TMPDIR="$R/tmp" PIP_CACHE_DIR="$R/cache/pip" CARGO_HOME="$R/cargo" RUSTUP_HOME="$R/rustup"
export CARGO_TARGET_DIR="$R/build/target-aarch64" CARGO_BUILD_JOBS=4 RUSTUP_TOOLCHAIN=1.96.1
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_DISABLE_PIP_VERSION_CHECK=1
"$IMAGE_PY" -I -m venv --without-pip "$R/env"
SITE=$("$R/env/bin/python" -I -c 'import sysconfig;print(sysconfig.get_path("purelib"))')
printf '%s\n' 'import site; site.addsitedir("/opt/sglang/lib/python3.12/site-packages")' > "$SITE/arm-image.pth"
export PATH="$R/env/bin:$CARGO_HOME/bin:$PATH"
"$IMAGE_PY" -I -m pip freeze --all > "$R/sources/image-freeze.txt"
python -I "$HERE/online-pins-receipt.py" pins "$PINS" "$R/sources/runtime-pins.txt"
python -I "$HERE/online-pins-receipt.py" requests "$PINS" "$R/sources/runtime-requests.txt"
printf '%s\n' 'maturin[patchelf]==1.9.6' 'patchelf==0.19.1.0' 'setuptools==84.0.0' 'wheel==0.48.0' 'setuptools-rust==1.11.1' 'setuptools-scm==9.2.2' 'hatchling==1.27.0' 'cmake==3.31.6' > "$R/sources/build-tool-pins.txt"
python -I -m pip download --only-binary=:all: -c "$R/sources/runtime-pins.txt" -r "$R/sources/runtime-requests.txt" -r "$R/sources/build-tool-pins.txt" --dest "$R/build/wheels" --extra-index-url https://download.pytorch.org/whl/cu130 --extra-index-url https://flashinfer.ai/whl --extra-index-url https://flashinfer.ai/whl/cu130
python -I -m pip install --no-index --find-links "$R/build/wheels" -c "$R/sources/runtime-pins.txt" -r "$R/sources/runtime-requests.txt" -r "$R/sources/build-tool-pins.txt" --report "$R/sources/runtime-install-report.json"
curl --proto '=https' --tlsv1.2 -fsSL https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init -o "$R/build/rustup-init"
curl --proto '=https' --tlsv1.2 -fsSL https://static.rust-lang.org/rustup/dist/aarch64-unknown-linux-gnu/rustup-init.sha256 -o "$R/build/rustup-init.sha256"
(cd "$R/build"; printf '%s  rustup-init\n' "$(awk '{print $1}' rustup-init.sha256)" | sha256sum --check -)
chmod +x "$R/build/rustup-init"
"$R/build/rustup-init" -y --no-modify-path --profile minimal --default-host aarch64-unknown-linux-gnu --default-toolchain 1.96.1
rustc -vV > "$R/sources/rustc.txt"
grep -qx 'release: 1.96.1' "$R/sources/rustc.txt"; grep -qx 'host: aarch64-unknown-linux-gnu' "$R/sources/rustc.txt"
git -C "$SRC" diff --binary HEAD > "$R/sources/dynamo.patch"
cp "$SRC/Cargo.lock" "$R/sources/Cargo.lock"
(cd "$SRC/lib/bindings/python"; "$R/env/bin/maturin" build --locked --release --features select-service,kv-indexer --target aarch64-unknown-linux-gnu --interpreter "$R/env/bin/python" --out "$R/build/wheels")
python -I -m pip wheel --no-deps --no-build-isolation "$SRC" --wheel-dir "$R/build/wheels"
python -I -m pip install --no-deps --ignore-installed "$R"/build/wheels/ai_dynamo-1.7.0-*.whl "$R"/build/wheels/ai_dynamo_runtime-1.7.0-*aarch64.whl
cmp "$SRC/Cargo.lock" "$R/sources/Cargo.lock"
python -I -m pip freeze --all > "$R/sources/installed-freeze.txt"
python -I "$HERE/online-pins-receipt.py" receipt "$PINS" "$R"
python -I -m pip check > "$R/sources/pip-check.txt" 2>&1
printf 'ARM_PR17655_BOOTSTRAP_COMPLETE\n'
