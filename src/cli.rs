use std::path::Path;

pub fn parse_flag(flag: &str, default: &str) -> String {
    let prefix = format!("{flag}=");
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        if arg == flag {
            if let Some(val) = args.next() {
                return val;
            }
        } else if let Some(rest) = arg.strip_prefix(&prefix) {
            return rest.to_string();
        }
    }
    default.to_string()
}

/// Directory holding the GGUF models, `tokenizer.json` and evaluation text.
pub fn assets_dir() -> String {
    let flag = parse_flag("--assets", "");
    let dir = if !flag.is_empty() {
        flag
    } else {
        match std::env::var("SQUIRREL_ASSETS") {
            Ok(dir) if !dir.is_empty() => dir,
            _ => "assets".to_string(),
        }
    };

    if !Path::new(&dir).is_dir() {
        panic!("Unable to find assets directory {dir:?}. Please use the --assets flag when running the binary or use the SQUIRREL_ASSETS environment variable to point to GGUF and tokenizer file.")
    }
    dir
}

/// Size the global rayon pool to the performance cores.
pub fn configure_thread_pool() {
    if std::env::var_os("RAYON_NUM_THREADS").is_some() {
        return;
    }
    let threads = performance_cores()
        .or_else(|| std::thread::available_parallelism().ok().map(|n| n.get()))
        .unwrap_or(1);
    let _ = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global();
}

/// Number of performance (P) cores on Apple Silicon, or `None` elsewhere.
#[cfg(target_os = "macos")]
fn performance_cores() -> Option<usize> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "hw.perflevel0.logicalcpu"])
        .output()
        .ok()?;
    String::from_utf8(out.stdout).ok()?.trim().parse().ok()
}

#[cfg(not(target_os = "macos"))]
fn performance_cores() -> Option<usize> {
    None
}
