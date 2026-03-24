use std::env;
use std::fs;
use std::path::PathBuf;

/// Write a custom config.cmake
fn generate_config_cmake() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
    let cmake_build_dir = out_dir.join("build");
    let _ = fs::create_dir_all(&cmake_build_dir);
    let path = cmake_build_dir.join("config.cmake");
    // Turn off Python bindings because we don't need and we don't want to rely on libpython.so.
    let contents = "set(XGRAMMAR_BUILD_PYTHON_BINDINGS OFF CACHE BOOL \"\" FORCE)\n";
    fs::write(&path, contents).expect("write config.cmake failed");
}

fn main() {
    generate_config_cmake();

    // Running the CMake build for the xgrammar library
    // Note: -Wno-deprecated-declarations is needed because xgrammar uses std::bind internally,
    // which triggers deprecation warnings for std::result_of in C++17 with newer compilers.
    // Note: -fno-lto is needed on AArch64, due to default linker differences on Rust among
    // x86_64 and others.
    // See https://blog.rust-lang.org/2025/09/01/rust-lld-on-1.90.0-stable/ for details.
    #[cfg(target_arch = "x86_64")]
    let cxx_flags: &str = "-Wno-deprecated-declarations";
    #[cfg(not(target_arch = "x86_64"))]
    let cxx_flags: &str = "-Wno-deprecated-declarations -fno-lto";

    let xgrammar = cmake::Config::new("thirdparty/xgrammar")
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
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/include/xgrammar/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/dlpack/include/dlpack/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/picojson/");
}
