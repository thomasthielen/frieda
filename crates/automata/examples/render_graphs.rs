//! Renders example automata to PNGs in `examples/graphs/` (also a quick manual smoke test for
//! local library changes).
//! Run with: cargo run --example render_graphs -p automata --features graphviz

use automata::TransitionSystem;
use automata::automaton::NCW;
use automata::core::Void;
use automata::dot::Dottable;
use automata::hoa::{HoaString, input::pop_omega_automaton};
use automata::ts::TSBuilder;
use std::path::PathBuf;

fn save_graph(name: &str, aut: &impl Dottable) {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/graphs");
    std::fs::create_dir_all(&dir).expect("failed to create graphs dir");
    let png = aut
        .render_graphviz()
        .expect("failed to render with graphviz");
    std::fs::write(dir.join(format!("{name}.png")), png).expect("failed to write png");
}

fn main() {
    let det1 = include_str!("hoa/det1.hoa");
    let nondet1 = include_str!("hoa/nondet1.hoa");

    let (det_aut, _) = pop_omega_automaton::<true>(HoaString::from(det1.to_string()))
        .expect("failed to parse det1.hoa");
    println!("parsed deterministic automaton with {} states", det_aut.ts().size());
    
    let (nondet_aut, _) = pop_omega_automaton::<false>(HoaString::from(nondet1.to_string()))
        .expect("failed to parse nondet1.hoa");
    println!("parsed nondeterministic automaton with {} states", nondet_aut.ts().size());

    // example automaton from RK22 - figure 1
    let nice_but_not_minimal = TSBuilder::without_state_colors()
        .with_edges([(0, 'a', false, 0), (0, 'b', true, 1), (1, 'a', false, 1), (1, 'b', true, 0)])
        .into_dcw(0); // 0 is the initial state
    save_graph("nice_but_not_minimal", &nice_but_not_minimal);

    // nondeterministic: two `a`-successors from state 0
    let ncw_test = NCW::from_parts(
        TSBuilder::<Void, bool, true>::without_state_colors()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_nts(),
        0,
    );
    save_graph("ncw", &ncw_test);

    // example automaton from RK22 - figure 2
    let safe_minimal_but_not_safe_centralized = TSBuilder::without_state_colors()
        .with_edges([(0, 'a', false, 0), (0, 'b', false, 1), (0, 'c', true, 2)])
        .with_edges([(1, 'c', false, 0), (1, 'a', true, 2), (1, 'b', true, 2)])
        .with_edges([(2, 'c', true, 0), (2, 'b', true, 1), (2, 'a', false, 2)])
        .into_dcw(0);
    save_graph(
        "safe_minimal_but_not_safe_centralized",
        &safe_minimal_but_not_safe_centralized,
    );
}