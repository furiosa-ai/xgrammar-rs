fn main() {
    // Running the CMake build for the xgrammar library
    let xgrammar =
        cmake::Config::new("thirdparty/xgrammar").define("CMAKE_CXX_COMPILER", "clang++").build();

    // Specifying the path to link the built library
    println!("cargo:rustc-link-search=native={}/build", xgrammar.display());
    println!("cargo:rustc-link-lib=static=xgrammar");

    // Setting the cpp build configuration
    let mut config = cpp_build::Config::new();
    config
        .include("thirdparty/xgrammar/include/")
        .include("thirdparty/xgrammar/3rdparty/dlpack/include/")
        .include("thirdparty/xgrammar/3rdparty/picojson/")
        .flag("-std=c++17");
    config.build("src/xgrammar/xgrammar.rs");

    // Re-run if any of these files change
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/xgrammar/xgrammar.rs");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/include/xgrammar/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/dlpack/include/dlpack/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/picojson/");
}
