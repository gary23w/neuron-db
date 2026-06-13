use neuron_core::Neuron;
use neuron_core::turn::turn;

fn n() -> Neuron { Neuron::new(1000) }

#[test]
fn statement_is_acked_then_recalled() {
    let mut m = n();
    let t = turn(&mut m, "the deploy region is us-west-2");
    assert_eq!(t.kind, "ack"); assert!(t.wrote >= 1);
    let q = turn(&mut m, "what is the deploy region?");
    assert_eq!(q.kind, "recall");
    assert!(q.reply.contains("us-west-2"), "{}", q.reply);
}

#[test]
fn unknown_question_is_idk() {
    let mut m = n();
    turn(&mut m, "the cache ttl is 300 seconds");
    let q = turn(&mut m, "what is the database password?");
    assert_eq!(q.kind, "idk", "{}", q.reply);
}

#[test]
fn arithmetic_symbols_and_words() {
    let mut m = n();
    assert_eq!(turn(&mut m, "2 + 3").kind, "math");
    assert!(turn(&mut m, "what is 2 + 3").reply.contains("= 5"));
    assert!(turn(&mut m, "what is 10 times 4").reply.contains("= 40"));
    assert!(turn(&mut m, "20 divided by 4").reply.contains("= 5"));
    assert!(turn(&mut m, "9 minus 14").reply.contains("= -5"));
}

#[test]
fn dates_are_not_arithmetic() {
    let mut m = n();
    let t = turn(&mut m, "the launch date is 2026-06-13");
    assert_eq!(t.kind, "ack", "a hyphenated date must store, not compute: {}", t.reply);
}

#[test]
fn greetings_and_self() {
    let mut m = n();
    assert_eq!(turn(&mut m, "hi").kind, "smalltalk");
    assert_eq!(turn(&mut m, "hiii").kind, "smalltalk");
    assert_eq!(turn(&mut m, "good morning").kind, "smalltalk");
    assert_eq!(turn(&mut m, "are you a real person?").kind, "self");
    assert_eq!(turn(&mut m, "what is your name?").kind, "self");
}

#[test]
fn yes_no_question() {
    let mut m = n();
    turn(&mut m, "my plan is pro");
    let t = turn(&mut m, "is my plan pro?");
    assert_eq!(t.kind, "recall");
    assert!(t.reply.to_lowercase().starts_with("yes") || t.reply.contains("pro"), "{}", t.reply);
}
