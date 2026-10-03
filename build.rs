use std::path::PathBuf;
use std::process::Command;

fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        return;
    }
    // screencapturekit 内部桥接了 Swift：运行时用系统自带的 /usr/lib/swift
    println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");
    // Swift 兼容库（swiftCompatibility56 等）在当前工具链目录里，
    // 只装了 Command Line Tools 时 crate 自己猜的 Xcode 路径不存在
    let swiftc = Command::new("xcrun").args(["--find", "swiftc"]).output();
    if let Ok(out) = swiftc
        && out.status.success()
    {
        let bin = PathBuf::from(String::from_utf8_lossy(&out.stdout).trim());
        if let Some(usr) = bin.parent().and_then(|b| b.parent()) {
            let lib = usr.join("lib/swift/macosx");
            if lib.exists() {
                println!("cargo:rustc-link-search=native={}", lib.display());
            }
        }
    }
}
