use std::process::Command;

fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();

    // No BLAS linking for WASM or other non-native targets.
    if target.contains("wasm") || target.contains("emscripten") {
        return;
    }

    if cfg!(target_os = "macos") {
        // Apple Accelerate — always available on macOS.
        println!("cargo:rustc-link-lib=framework=Accelerate");
        println!("cargo:rustc-cfg=has_blas");
    } else if cfg!(target_os = "linux") {
        if has_lib("openblas") {
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:rustc-cfg=has_blas");
        } else if has_lib("blas") {
            println!("cargo:rustc-link-lib=dylib=blas");
            println!("cargo:rustc-cfg=has_blas");
        }
    } else if cfg!(target_os = "windows") {
        if has_lib("openblas") {
            println!("cargo:rustc-link-lib=dylib=openblas");
            println!("cargo:rustc-cfg=has_blas");
        }
    }
}

fn has_lib(name: &str) -> bool {
    if let Ok(status) = Command::new("pkg-config")
        .args(["--exists", name])
        .status()
    {
        if status.success() {
            if let Ok(output) = Command::new("pkg-config")
                .args(["--libs", name])
                .output()
            {
                let flags = String::from_utf8_lossy(&output.stdout);
                for flag in flags.split_whitespace() {
                    if let Some(dir) = flag.strip_prefix("-L") {
                        println!("cargo:rustc-link-search=native={}", dir);
                    }
                }
            }
            return true;
        }
    }

    let lib_dirs: &[&str] = if cfg!(target_os = "linux") {
        &[
            "/usr/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib/aarch64-linux-gnu",
            "/usr/lib64",
        ]
    } else {
        &[]
    };

    let ext = if cfg!(target_os = "windows") { "lib" } else { "so" };
    let so_name = format!("lib{}.{}", name, ext);

    for dir in lib_dirs {
        if std::path::Path::new(dir).join(&so_name).exists() {
            return true;
        }
    }

    false
}
