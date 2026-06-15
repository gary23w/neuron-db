use neuron_core::Neuron;
fn val(n:&mut Neuron,q:&str)->Option<String>{ n.recall(q).map(|r| r.value) }

#[test] fn basic_recall(){ let mut n=Neuron::new(500);
    n.observe("my name is Marisol"); n.observe("the wifi password is hunter2");
    assert_eq!(val(&mut n,"what is my name?").as_deref(), Some("Marisol"));
    assert_eq!(val(&mut n,"what is the wifi password?").as_deref(), Some("hunter2")); }

#[test] fn value_isolation(){ let mut n=Neuron::new(500);
    n.observe("only the first 1,000 users score 150,000 coins");
    assert_eq!(val(&mut n,"how many users?").as_deref(), Some("1,000"));
    assert_eq!(val(&mut n,"how many coins?").as_deref(), Some("150,000")); }

#[test] fn abstention(){ let mut n=Neuron::new(500);
    n.observe("my name is Gary");
    assert!(n.recall("what is my blood type?").is_none()); }

#[test] fn relation_binding(){ let mut n=Neuron::new(500);
    n.observe("my dog is called Biscuit"); n.observe("my sister is called Dana");
    assert_eq!(val(&mut n,"what is my dog's name?").as_deref(), Some("Biscuit"));
    assert_eq!(val(&mut n,"what is my sister's name?").as_deref(), Some("Dana")); }

#[test] fn distinct_keys_scale(){ let mut n=Neuron::new(2000); let mut p=Vec::new();
    let adjs=["north","south","east","west","main","spare","old","new","blue","red"];
    let nouns=["wifi","door","bank","email","garage","locker","safe","router","vault","gate"];
    let things=["password","code","pin","key"]; let mut i=0;
    for a in adjs { for no in nouns { for th in things {
        if i>=400 {break;} let v=format!("{}{}",1000+i,(b'A'+(i%26)as u8)as char);
        n.observe(&format!("the {} {} {} is {}",a,no,th,v)); p.push((format!("what is the {} {} {}?",a,no,th),v)); i+=1; }}}
    let hits=p.iter().filter(|(q,a)| n.recall(q).is_some_and(|r| &r.value==a)).count();
    assert_eq!(hits, p.len(), "distinct keys should recall 100%"); }

#[test] fn dump_load_roundtrip(){ let mut n=Neuron::new(500);
    n.observe("my name is Marisol"); n.observe("the door code is 4452");
    let blob=n.dump(); let mut n2=Neuron::load(&blob,500);
    assert_eq!(val(&mut n2,"what is my name?").as_deref(), Some("Marisol"));
    assert_eq!(val(&mut n2,"what is the door code?").as_deref(), Some("4452")); }

#[test] fn number_before_noun(){ let mut n=Neuron::new(500);
    n.observe("the room holds 50 chairs, 8 tables and 200 guests");
    assert_eq!(val(&mut n,"how many guests?").as_deref(), Some("200"));
    assert_eq!(val(&mut n,"how many chairs?").as_deref(), Some("50"));
    assert_eq!(val(&mut n,"how many tables?").as_deref(), Some("8")); }

#[test]
fn semantic_aliases_bridge_paraphrase() {
    let mut n = neuron_core::Neuron::new(500);
    n.observe("my plan is pro");
    n.observe("my manager is Dana");
    assert_eq!(n.recall("what subscription tier do i have?").unwrap().value, "pro");
    assert_eq!(n.recall("who do i report to?").unwrap().value, "Dana");
}
