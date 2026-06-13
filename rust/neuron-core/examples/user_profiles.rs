//! Per-user memory for an app/website backend. One scope per user; scopes are isolated.
//! Run: cargo run --release --example user_profiles --features sqlite
use neuron_core::db::NeuronDB;

fn main() {
    let db = NeuronDB::open("/tmp/neuron_profiles.db", 500);

    // onboard a few users
    for (uid, plan, theme) in [("u:1", "pro", "dark"), ("u:2", "free", "light"), ("u:3", "enterprise", "dark")] {
        db.observe(uid, &format!("the plan is {}", plan));
        db.observe(uid, &format!("the theme is {}", theme));
    }

    // per-user lookups (microseconds, straight from the file)
    for uid in ["u:1", "u:2", "u:3"] {
        println!("{}: plan={:?} theme={:?}", uid,
            db.get(uid, "what plan?"), db.get(uid, "what theme?"));
    }

    // scopes don't bleed: u:1's data never shows up under u:2
    assert_ne!(db.get("u:1", "what plan?"), db.get("u:2", "what plan?"));
    println!("\nall scopes: {:?}", db.neurons());
}
