fn main() {
    // Building XGrammar C++ library and linking it to Rust
    let xgrammar = cmake::Config::new("thirdparty/xgrammar")
    .define("CMAKE_CXX_COMPILER", "clang++")
    .build();

    println!("cargo:rustc-link-search=native={}/build", xgrammar.display());
    println!("cargo:rustc-link-lib=static=xgrammar");
    println!("cargo::warning=Linking XGrammar C++ library from {}/build", xgrammar.display());

    let exist = std::fs::exists(format!("{}/build/libxgrammar.a", xgrammar.display()))
        .expect("failed to check existence of libxgrammar.a");
    assert!(exist, "libxgrammar.a does not exist in the build directory");

    let mut config = cpp_build::Config::new();
    config
        .include("thirdparty/xgrammar/include/")
        .include("thirdparty/xgrammar/3rdparty/dlpack/include/")
        .include("thirdparty/xgrammar/3rdparty/picojson/")
        .flag("-std=c++17");
    config.build("src/xgrammar/tokenizer_info.rs");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/xgrammar/tokenizer_info.rs");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/include/xgrammar/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/dlpack/include/dlpack/");
    println!("cargo:rerun-if-changed=thirdparty/xgrammar/3rdparty/picojson/");
}
