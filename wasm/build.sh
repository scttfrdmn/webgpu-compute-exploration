#!/bin/bash

set -e
cd "$(dirname "$0")"

echo "Building WASM module..."

if ! command -v wasm-pack &> /dev/null; then
    echo "Error: wasm-pack not found. Install with: cargo install wasm-pack"
    exit 1
fi

# wasm-pack uses whichever rustc is first on PATH. A non-rustup rustc (e.g. Homebrew)
# usually has no wasm32 target and fails with an opaque "not found in sysroot" error.
if [ ! -d "$(rustc --print sysroot)/lib/rustlib/wasm32-unknown-unknown" ]; then
    echo "Error: $(command -v rustc) has no wasm32-unknown-unknown target."
    echo "  If you use rustup:  rustup target add wasm32-unknown-unknown"
    echo "  If rustc is from Homebrew, put rustup's toolchain first:  export PATH=\"\$HOME/.cargo/bin:\$PATH\""
    exit 1
fi

wasm-pack build --target web --out-dir ../js/wasm-pkg

echo "Build complete! WASM module is ready at js/wasm-pkg/"
