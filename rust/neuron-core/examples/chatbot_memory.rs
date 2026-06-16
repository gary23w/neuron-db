//! Wire neuron-db in as a chatbot's long-term memory. The pattern, every turn:
//!   1. WRITE  — store durable facts the user states (so they survive across sessions).
//!   2. RECALL — before calling your LLM, pull the few facts relevant to the new message.
//!   3. INJECT — prepend those facts to the model's context so it answers with memory.
//!
//! The store stays microseconds and deterministic; the LLM only ever sees a tiny, relevant
//! memory block, never the whole database.
//!
//! Run: cargo run --release --example chatbot_memory --features sqlite
use neuron_core::db::NeuronDB;

/// Build the memory block you prepend to the LLM context for this user + message.
fn recall_context(db: &NeuronDB, user: &str, message: &str) -> String {
    // Pull the single most relevant fact (use recall_related / multiple calls for more).
    match db.recall(user, message) {
        // tune this threshold: higher = only very relevant memory is injected
        Some(hit) if hit.coverage >= 0.5 => format!("[memory] {}\n", hit.fact),
        _ => String::new(),
    }
}

/// Stand-in for "call your LLM with a system prompt + memory + the user message".
fn llm_reply(memory: &str, message: &str) -> String {
    if memory.is_empty() { format!("(no memory) you said: {}", message) }
    else { format!("using memory -> {}| answering: {}", memory.trim(), message) }
}

fn main() {
    let db = NeuronDB::open(&std::env::temp_dir().join("neuron_chatbot.db").to_string_lossy(), 500);
    let user = "user:alice";

    // A conversation. Statements get remembered; questions recall + inject.
    let convo = [
        "my name is Alice",
        "i work as a data engineer",
        "my timezone is PST",
        "what's my name?",          // <- should recall "Alice"
        "what is my timezone?",     // <- should recall "PST"
    ];

    for msg in convo {
        // 1. RECALL relevant memory for the incoming message, using what we knew BEFORE it.
        let memory = recall_context(&db, user, msg);

        // 2. INJECT into the model context and answer.
        let reply = llm_reply(&memory, msg);
        println!("user> {}\nbot>  {}\n", msg, reply);

        // 3. WRITE any durable facts the user stated (skip questions; turn() routes this too).
        if !msg.contains('?') { db.observe(user, msg); }
    }

    // Or let neuron-db do the store-or-answer routing in ONE call with turn():
    println!("--- one-call routing with turn() ---");
    for msg in ["my favorite editor is neovim", "what is my favorite editor?"] {
        println!("user> {}\nbot>  {}\n", msg, db.turn(user, msg).reply);
    }
}
