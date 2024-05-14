fn main() {
    println!("cargo:rustc-link-lib=dylib=lapacke");
    println!("cargo:rustc-link-lib=dylib=lapack");
    println!("cargo:rustc-link-search=native=/path/to/your/lapack/lib");
}
