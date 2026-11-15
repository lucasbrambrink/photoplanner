fn main() {
    println!("cargo:rerun-if-changed=src/metal_bridge.m");
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        #[cfg(target_os = "macos")]
        {
            cc::Build::new()
                .file("src/metal_bridge.m")
                .flag("-fobjc-arc")
                .flag("-O3")
                .compile("pp_metal_bridge");
            println!("cargo:rustc-link-lib=framework=Metal");
            println!("cargo:rustc-link-lib=framework=Foundation");
        }
    }
}
