//! Laboratory: a general testing environment for experimenting with the library (e.g. the
//! minimization algorithm). Rendered automata are written as PNGs to `examples/graphs/`.
//! Run with: cargo run --example laboratory -p automata --features graphviz

use automata::automaton::{DCW, NCW};
use automata::core::Void;
use automata::core::alphabet::CharAlphabet;
use automata::dot::Dottable;
use automata::hoa::{HoaString, input::pop_omega_automaton};
use automata::ts::TSBuilder;
use automata::{Pointed, TransitionSystem};
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
    println!(
        "parsed deterministic automaton with {} states",
        det_aut.ts().size()
    );

    let (nondet_aut, _) = pop_omega_automaton::<false>(HoaString::from(nondet1.to_string()))
        .expect("failed to parse nondet1.hoa");
    println!(
        "parsed nondeterministic automaton with {} states",
        nondet_aut.ts().size()
    );

    // example automaton from AK22 - figure 1
    let nice_but_not_minimal = TSBuilder::without_state_colors()
        .with_edges([
            (0, 'a', false, 0),
            (0, 'b', true, 1),
            (1, 'a', false, 1),
            (1, 'b', true, 0),
        ])
        .into_dcw(0); // 0 is the initial state
    save_graph("nice_but_not_minimal", &nice_but_not_minimal);
    assert!(nice_but_not_minimal.is_safe_deterministic());

    // nondeterministic: two `a`-successors from state 0
    let ncw_test = NCW::from_parts(
        TSBuilder::<Void, bool, true>::without_state_colors()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_nts(),
        0,
    );
    save_graph("ncw", &ncw_test);
    assert!(ncw_test.is_safe_deterministic());

    // example automaton from AK22 - figure 2
    let safe_minimal_but_not_safe_centralized = DCW::builder()
        .with_edges([
            (0, 'a', false, 0),
            (0, 'b', false, 1),
            (0, 'c', true, 2),
            (1, 'a', true, 2),
            (1, 'b', true, 2),
            (1, 'c', false, 0),
            (2, 'a', false, 2),
            (2, 'b', true, 1),
            (2, 'c', true, 0),
        ])
        .into_dcw(0);
    save_graph(
        "safe_minimal_but_not_safe_centralized",
        &safe_minimal_but_not_safe_centralized,
    );

    // example automaton from AK22 - figure 4
    let safe_centralized_result = NCW::builder()
        .with_edges([
            (0, 'a', false, 0),
            (0, 'b', false, 1),
            (1, 'c', false, 0),
            (0, 'c', true, 0),
            (0, 'c', true, 1),
            (1, 'a', true, 1),
            (1, 'b', true, 1),
            (1, 'a', true, 0),
            (1, 'b', true, 0),
        ])
        .into_ncw(0);
    assert!(safe_centralized_result.is_safe_deterministic());

    // semantic determinization
    let mut ncw = NCW::builder()
        .with_edges([
            (0, 'a', true, 1),
            (0, 'a', true, 2),
            (0, 'a', false, 3),
            (0, 'b', false, 0),
            (1, 'a', true, 1),
            (1, 'b', false, 1),
            (2, 'a', false, 2),
            (2, 'b', false, 2),
            (3, 'a', true, 3),
            (3, 'b', true, 3),
        ])
        .into_ncw(0);
    save_graph("ncw_test", &ncw);
    ncw.semantically_determinize();
    save_graph("ncw_test_semantically_determinized", &ncw);

    let example_check_01 = DCW::builder()
        .with_edges([
            (0, 'a', true, 1),
            (0, 'b', false, 0),
            (1, 'a', true, 1),
            (1, 'b', false, 0),
        ])
        .into_dcw(0);
    save_graph("example_check_01", &example_check_01);

    let determinize = DCW::builder()
        .with_edges([
            (0, 'a', false, 0),
            (0, 'b', true, 1),
            (1, 'a', false, 1),
            (1, 'b', true, 0),
        ])
        .into_dcw(0);
    save_graph("determinize", &determinize);

    full_pipeline_example();
}

/// Returns the transitions of `ts` as a sorted list, for comparing against expectations.
fn sorted_transitions(
    ts: &impl TransitionSystem<StateIndex = u32, Alphabet = CharAlphabet, EdgeColor = bool>,
) -> Vec<(u32, char, bool, u32)> {
    let mut transitions: Vec<_> = ts
        .state_indices()
        .flat_map(|q| ts.transitions_from(q).collect::<Vec<_>>())
        .collect();
    transitions.sort();
    transitions
}

/// An example in which every step of the minimization pipeline of AK22 has an effect.
///
/// The language over {a, b, c} is "eventually, every `b` is immediately followed by an `a`",
/// i.e. the factors `bb` and `bc` (the *violations*) occur only finitely often. It is
/// prefix-independent, so every state with a nonempty language accepts exactly it.
///
/// States:
/// - q1, q3: two copies of "the last letter was not a `b`". They read `a` and `c` safely
///   (`c` switches between the copies) and go safely to their copy of "the last letter was a
///   `b`" on `b` (q1 to q2, q3 to q4).
/// - q2, q4: two copies of "the last letter was a `b`". They read `a` safely into the *other*
///   copy (q2 to q3, q4 to q1), so that {q1, q2, q3, q4} is one safe component; `b` and `c`
///   are violations, i.e. `α`-transitions.
/// - q0 (initial): a pre-state that reads `a` safely in a self-loop, `b` as an `α`-transition
///   to q2 and `c` *safely* to q1.
/// - q5: a rejecting sink (empty language), reached only by an extra `α`-transition
///   ⟨q0, b, q5⟩.
///
/// Without ⟨q0, b, q5⟩ this is a DCW. The extra transition makes it a (safe deterministic)
/// NCW, but it stays GFG: a strategy that never takes ⟨q0, b, q5⟩ simply runs the DCW.
/// Adding ⟨q0, b, q5⟩ also does not change the language, as `L(q5) = ∅`.
///
/// What each step does:
/// 1. semantic determinization: ⟨q0, b, q5⟩ is not covering (`L(q5) = ∅ ⊊ L(q2)`), so it is
///    removed,
/// 2. removing unreachable states: q5 is now unreachable and removed,
/// 3. normalization: the safe transition ⟨q0, c, q1⟩ connects the safe components {q0} and
///    {q1, q2, q3, q4}, so it becomes an `α`-transition,
/// 4. safe centralization: `L_safe(q0) = a^ω ⊆ L_safe(q1)`, but not vice versa, so {q0} is not
///    in the frontier and removed; the initial state moves to q1. All kept states are
///    equivalent, so every `α`-transition is redirected to all four of them, which makes the
///    automaton nondeterministic,
/// 5. safe minimization: q1 ≈ q3 and q2 ≈ q4 (the copies are symmetric), so they are merged.
///
/// The result has two states, q1 and q2, with the `α`-transitions on `b` and `c` leading
/// nondeterministically to both. It is minimal, as a single state cannot recognize the
/// language: it would need safe self-loops on `b` (for `(ba)^ω`) and `c` (for `c^ω`), and then
/// also accept `b^ω`.
fn full_pipeline_example() {
    const INPUT: [(u32, char, bool, u32); 19] = [
        // q0
        (0, 'a', false, 0),
        (0, 'b', true, 2),
        (0, 'b', true, 5),
        (0, 'c', false, 1),
        // q1
        (1, 'a', false, 1),
        (1, 'b', false, 2),
        (1, 'c', false, 3),
        // q2
        (2, 'a', false, 3),
        (2, 'b', true, 2),
        (2, 'c', true, 1),
        // q3
        (3, 'a', false, 3),
        (3, 'b', false, 4),
        (3, 'c', false, 1),
        // q4
        (4, 'a', false, 1),
        (4, 'b', true, 4),
        (4, 'c', true, 3),
        // q5
        (5, 'a', true, 5),
        (5, 'b', true, 5),
        (5, 'c', true, 5),
    ];
    let mut ncw = NCW::builder().with_edges(INPUT).into_ncw(0);
    save_graph("pipeline_0_input", &ncw);
    assert!(ncw.is_safe_deterministic()); // the only nondeterministic choice is on `b` from q0
    assert!(!ncw.is_semantically_deterministic());

    // 1. semantic determinization removes the non-covering ⟨q0, b, q5⟩, as L(q5) = ∅ ⊊ L(q2)
    ncw.semantically_determinize();
    save_graph("pipeline_1_semantically_determinized", &ncw);
    assert!(ncw.is_semantically_deterministic());
    assert!(!sorted_transitions(&ncw).contains(&(0, 'b', true, 5)));
    assert_eq!(sorted_transitions(&ncw).len(), INPUT.len() - 1);

    // 2. q5 is now unreachable and thus removed
    let removed: Vec<u32> = ncw
        .remove_unreachable_states()
        .into_iter()
        .map(|(q, _)| q)
        .collect();
    save_graph("pipeline_2_unreachable_removed", &ncw);
    assert_eq!(removed, vec![5]);

    // 3. ⟨q0, c, q1⟩ connects different safe components, thus it is set to true
    assert!(!ncw.is_normal());
    ncw.normalize();
    save_graph("pipeline_3_normalized", &ncw);
    assert!(ncw.is_normal());
    assert!(sorted_transitions(&ncw).contains(&(0, 'c', true, 1)));
    assert!(!sorted_transitions(&ncw).contains(&(0, 'c', false, 1)));

    // 4. {q0} is not in the frontier, thus it is removed
    assert!(!ncw.is_safe_centralized());
    ncw.safe_centralize();
    save_graph("pipeline_4_safe_centralized", &ncw);
    assert!(ncw.is_safe_centralized());
    assert!(ncw.is_alpha_homogeneous()); // side effect of safe centralization
    assert_eq!(ncw.initial(), 1); // the initial state moved to q1
    assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![1, 2, 3, 4]);
    let all_p_r = [1, 2, 3, 4];
    for (source, letter) in [(2, 'b'), (2, 'c'), (4, 'b'), (4, 'c')] {
        for target in all_p_r {
            assert!(sorted_transitions(&ncw).contains(&(source, letter, true, target)));
        }
    }

    // 5. q1 ≈ q3 and q2 ≈ q4 are merged
    assert!(!ncw.is_safe_minimal());
    assert_eq!(
        ncw.strong_equivalence_relation()
            .into_iter()
            .filter(|(q, s)| q < s)
            .collect::<Vec<_>>(),
        vec![(1, 3), (2, 4)]
    );
    ncw.safe_minimize();
    save_graph("pipeline_5_safe_minimized", &ncw);

    let expected = vec![
        (1, 'a', false, 1),
        (1, 'b', false, 2),
        (1, 'c', false, 1),
        (2, 'a', false, 1),
        (2, 'b', true, 1),
        (2, 'b', true, 2),
        (2, 'c', true, 1),
        (2, 'c', true, 2),
    ];
    assert_eq!(ncw.initial(), 1);
    assert_eq!(sorted_transitions(&ncw), expected);
    // all required properties are satisfied
    assert!(ncw.is_safe_deterministic());
    assert!(ncw.is_semantically_deterministic());
    assert!(ncw.is_normal());
    assert!(ncw.is_safe_centralized());
    assert!(ncw.is_alpha_homogeneous());
    assert!(ncw.is_safe_minimal());

    // running the whole pipeline in one go gives the same result
    let mut minimized = NCW::builder().with_edges(INPUT).into_ncw(0);
    minimized.minimize();
    assert_eq!(minimized.initial(), 1);
    assert_eq!(sorted_transitions(&minimized), expected);
    println!("full pipeline example: minimized 6 states to 2");
}
