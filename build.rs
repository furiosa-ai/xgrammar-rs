use std::env;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

/// Write a custom config.cmake
fn generate_config_cmake(out_dir: &Path) {
    let cmake_build_dir = out_dir.join("build");
    fs::create_dir_all(&cmake_build_dir).expect("create CMake build directory failed");
    let path = cmake_build_dir.join("config.cmake");
    // Turn off Python bindings because we don't need and we don't want to rely on libpython.so.
    let contents = "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL \"\" FORCE)\n";
    fs::write(&path, contents).expect("write config.cmake failed");
}

fn main() {
    // Cargo's package verification can reuse OUT_DIR for a different source path.
    // Give each source its own CMake directory so cmake-rs doesn't clear the
    // config.cmake we just wrote when it detects a changed CMAKE_HOME_DIRECTORY.
    let mut source_hash = DefaultHasher::new();
    fs::canonicalize("thirdparty/xgrammar")
        .expect("xgrammar source missing")
        .hash(&mut source_hash);
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"))
        .join(format!("xgrammar-{:x}", source_hash.finish()));
    generate_config_cmake(&out_dir);

    // Running the CMake build for the xgrammar library
    // Note: -Wno-deprecated-declarations is needed because xgrammar uses std::bind internally,
    // which triggers deprecation warnings for std::result_of in C++17 with newer compilers.
    // Note: -fno-lto is needed on AArch64, due to default linker differences on Rust among
    // x86_64 and others.
    // See https://blog.rust-lang.org/2025/09/01/rust-lld-on-1.90.0-stable/ for details.
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").expect("CARGO_CFG_TARGET_ARCH is not set");
    let cxx_flags = if target_arch == "x86_64" {
        "-Wno-deprecated-declarations"
    } else {
        "-Wno-deprecated-declarations -fno-lto"
    };

    let xgrammar = cmake::Config::new("thirdparty/xgrammar")
        .out_dir(&out_dir)
        .define("CMAKE_CXX_COMPILER", "clang++")
        .define("CMAKE_CXX_FLAGS", cxx_flags)
        .build_target("xgrammar")
        .build();

    // Specifying the path to link the built library
    println!("cargo:rustc-link-search=native={}/build", xgrammar.display());
    println!("cargo:rustc-link-lib=static=xgrammar");

    // Setting the cpp build configuration
    let mut config = cpp_build::Config::new();
    config
        .include("thirdparty/xgrammar/include/")
        // List all thirdparty libraries' headers
        .include("thirdparty/xgrammar/3rdparty/dlpack/include/")
        .include("thirdparty/xgrammar/3rdparty/picojson/")
        .flag("-std=c++17");
    config.build("src/lib.rs");

    // Re-run if any of these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/lib.rs");
    // Upstream releases can change C++ sources without changing public headers.
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/CMakeLists.txt");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/cpp/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/include/xgrammar/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/dlpack/include/dlpack/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/picojson/");
}
