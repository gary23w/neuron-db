#![cfg(all(feature = "sqlite", feature = "cortex"))]

use std::{
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

struct Store(PathBuf);

impl Store {
    fn new() -> Self {
        let stamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        Self(std::env::temp_dir().join(format!(
            "neuron-stats-batch-{}-{}.db",
            std::process::id(),
            stamp
        )))
    }

    fn run(&self, args: &[&str]) -> String {
        let out = Command::new(env!("CARGO_BIN_EXE_neuron"))
            .arg("--db")
            .arg(&self.0)
            .args(args)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
        String::from_utf8(out.stdout).unwrap()
    }

    fn count(&self, scope: &str) -> usize {
        let out = self.run(&["--json", "stats", scope]);
        out.split("\"facts\":")
            .nth(1)
            .unwrap()
            .split(',')
            .next()
            .unwrap()
            .parse()
            .unwrap()
    }
}

impl Drop for Store {
    fn drop(&mut self) {
        for suffix in ["", "-wal", "-shm"] {
            let _ = std::fs::remove_file(format!("{}{}", self.0.display(), suffix));
        }
    }
}

#[test]
fn batch_preserves_scope_order_duplicates_and_the_single_scope_counts() {
    let store = Store::new();
    store.run(&[
        "observe",
        "lessons",
        "Run the formatter before the linter so the linter sees final source code",
    ]);
    store.run(&[
        "observe",
        "facts",
        "The release uses Zig version 0.16 and ships a native desktop executable",
    ]);
    let lessons = store.count("lessons");
    let facts = store.count("facts");
    assert!(lessons > 0 && facts > 0);
    let actual = store.run(&[
        "--json",
        "stats-batch",
        "facts",
        "missing",
        "lessons",
        "facts",
    ]);
    assert_eq!(actual, format!("{{\"scopes\":[{{\"scope\":\"facts\",\"facts\":{facts}}},{{\"scope\":\"missing\",\"facts\":0}},{{\"scope\":\"lessons\",\"facts\":{lessons}}},{{\"scope\":\"facts\",\"facts\":{facts}}}]}}\n"));
}

#[test]
fn batch_escapes_scope_names_and_preserves_stored_facts() {
    let store = Store::new();
    let scope = "quoted\"scope\\name";
    store.run(&[
        "observe",
        scope,
        "The deployment keeps the original source files available for inspection",
    ]);
    let before = store.run(&["export", scope]);
    let count = store.count(scope);
    assert!(count > 0);
    let actual = store.run(&["--json", "stats-batch", scope]);
    assert_eq!(
        actual,
        format!("{{\"scopes\":[{{\"scope\":\"quoted\\\"scope\\\\name\",\"facts\":{count}}}]}}\n")
    );
    assert_eq!(before, store.run(&["export", scope]));
}

#[test]
fn batch_requires_at_least_one_scope() {
    let store = Store::new();
    let out = Command::new(env!("CARGO_BIN_EXE_neuron"))
        .arg("--db")
        .arg(&store.0)
        .args(["--json", "stats-batch"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(out.stdout.is_empty());
}
