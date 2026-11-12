#![cfg(all(feature = "sqlite", feature = "cortex"))]
//! Keyed facts through the `neuron` CLI: upsert replaces by exact key, forgetkey removes by exact key.

use std::{
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

struct Store(PathBuf);

impl Store {
    fn new() -> Self {
        let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        Self(std::env::temp_dir().join(format!("neuron-keyed-cli-{}-{}.db", std::process::id(), stamp)))
    }
    fn raw(&self, args: &[&str]) -> Output {
        Command::new(env!("CARGO_BIN_EXE_neuron")).arg("--db").arg(&self.0).args(args).output().unwrap()
    }
    fn run(&self, args: &[&str]) -> String {
        let out = self.raw(args);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        String::from_utf8(out.stdout).unwrap().trim().to_string()
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
fn upsert_replaces_and_forgetkey_removes_by_exact_key() {
    let s = Store::new();
    assert_eq!(s.run(&["--json", "upsert", "prefs", "editor", "the preferred editor is Helix"]), "{\"wrote\":1}");
    assert_eq!(s.run(&["--json", "upsert", "prefs", "editor2", "the backup editor is Nano"]), "{\"wrote\":1}");
    // each process reopens the store, so the key has to survive on disk for the replace to work
    assert_eq!(s.run(&["upsert", "prefs", "editor", "the", "preferred", "editor", "is", "Zed"]), "stored 1 fact(s)");
    let hits = s.run(&["recall", "prefs", "what is the preferred editor"]);
    assert!(hits.contains("Zed") && !hits.contains("Helix"), "the old value must be gone: {hits}");
    assert_eq!(s.run(&["--json", "forgetkey", "prefs", "editor"]), "{\"forgot\":1,\"remaining\":1}", "editor must not match editor2");
    assert_eq!(s.run(&["forgetkey", "prefs", "editor"]), "forgot 0, 1 remaining");
    assert!(s.run(&["recall", "prefs", "what is the backup editor"]).contains("Nano"));
}

#[test]
fn a_missing_key_or_text_is_a_usage_error_and_changes_nothing() {
    let s = Store::new();
    s.run(&["observe", "prefs", "the preferred editor is Helix"]);
    for args in [&["forgetkey", "prefs"][..], &["forgetkey", "prefs", "a", "b"][..], &["upsert", "prefs", "editor"][..], &["upsert", "prefs"][..]] {
        let out = s.raw(args);
        assert_eq!(out.status.code(), Some(2), "{args:?}");
        assert!(String::from_utf8_lossy(&out.stderr).contains("usage:"), "{args:?}");
    }
    assert!(s.run(&["recall", "prefs", "what is the preferred editor"]).contains("Helix"), "a keyless forgetkey must not wipe the scope");
}
