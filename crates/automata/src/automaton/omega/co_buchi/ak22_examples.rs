//! Examples from \[AK22\] and further automata shared by the tests of the co-Büchi modules.
//!
//! \[AK22\] Bader Abu Radi, Orna Kupferman: *Minimization and Canonization of GFG
//! Transition-Based Automata*. Log. Methods Comput. Sci. 18(3) (2022)
use super::{DCW, NCW};
use crate::TransitionSystem;
use automata_core::alphabet::CharAlphabet;

/// AK22, Figure 1: the tDCW `A_fm` for "finitely many `b`s". Both states are strongly-equivalent
/// with safe language `a^ω`, but lie in different safe components `{q0}` and `{q1}`.
pub(super) const AK22_FIGURE_1: [(u32, char, bool, u32); 4] = [
    (0, 'a', false, 0),
    (0, 'b', true, 1),
    (1, 'a', false, 1),
    (1, 'b', true, 0),
];

pub(super) fn ak22_figure_1() -> DCW {
    DCW::builder().with_edges(AK22_FIGURE_1).into_dcw(0)
}

/// AK22, Figure 2: a nice tDCW whose states are all equivalent, but differ in their safe
/// languages. Its safe components are `{q0, q1}` and `{q2}`.
pub(super) const AK22_FIGURE_2: [(u32, char, bool, u32); 9] = [
    (0, 'a', false, 0),
    (0, 'b', false, 1),
    (0, 'c', true, 2),
    (1, 'a', true, 2),
    (1, 'b', true, 2),
    (1, 'c', false, 0),
    (2, 'a', false, 2),
    (2, 'b', true, 1),
    (2, 'c', true, 0),
];

pub(super) fn ak22_figure_2() -> DCW {
    DCW::builder().with_edges(AK22_FIGURE_2).into_dcw(0)
}

/// AK22, Figure 2 with additional states, such that every minimization step has an effect:
/// - the `α`-transition `⟨q0, a, q3⟩` into the empty state 3 is not covering, and removing it
///   makes 3 unreachable, just like 4,
/// - the initial state 5 reads every letter safely into q0, connecting the safe components
///   `{q5}` and `{q0, q1}`, so these transitions are recolored during normalization,
/// - safe centralization then removes q5 and q2, and moves the initial state to q0.
pub(super) fn ak22_figure_2_extended() -> NCW {
    NCW::builder()
        .with_edges(AK22_FIGURE_2)
        .with_edges([
            (0, 'a', true, 3),
            (3, 'a', true, 3),
            (3, 'b', true, 3),
            (3, 'c', true, 3),
            (4, 'a', true, 4),
            (5, 'a', false, 0),
            (5, 'b', false, 0),
            (5, 'c', false, 0),
        ])
        .into_ncw(5)
}

/// The tNCW `B_S` for `S = {{q0, q1}}` of AK22, Figure 4, obtained from Figure 2.
pub(super) const AK22_FIGURE_4: [(u32, char, bool, u32); 9] = [
    (0, 'a', false, 0),
    (0, 'b', false, 1),
    (0, 'c', true, 0),
    (0, 'c', true, 1),
    (1, 'a', true, 0),
    (1, 'a', true, 1),
    (1, 'b', true, 0),
    (1, 'b', true, 1),
    (1, 'c', false, 0),
];

/// State 0 reads `a` safely into the universal state 1, and via the `α`-transition `⟨0, a, 2⟩`
/// into the empty state 2. The latter is not covering, so this is safe deterministic and GFG,
/// but not semantically deterministic.
pub(super) fn non_covering_ncw() -> NCW {
    NCW::builder()
        .with_edges([
            (0, 'a', false, 1),
            (0, 'a', true, 2),
            (0, 'b', false, 0),
            (1, 'a', false, 1),
            (1, 'b', false, 1),
            (2, 'a', true, 2),
            (2, 'b', true, 2),
        ])
        .into_ncw(0)
}

/// State 0 has two safe transitions on `a`, so this is not safe deterministic.
pub(super) fn not_safe_deterministic_ncw() -> NCW {
    NCW::builder()
        .with_edges([
            (0, 'a', false, 1),
            (0, 'a', false, 2),
            (1, 'a', false, 1),
            (2, 'a', false, 2),
        ])
        .into_ncw(0)
}

/// State 0 has both a safe and an `α`-transition on `a`, so this is safe deterministic, but not
/// `α`-homogenous.
pub(super) fn safe_and_alpha_on_same_letter_ncw() -> NCW {
    NCW::builder()
        .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
        .into_ncw(0)
}

/// Returns the transitions of `ncw` as a sorted list, for comparing against expectations.
pub(super) fn sorted_transitions<
    T: TransitionSystem<StateIndex = u32, Alphabet = CharAlphabet, EdgeColor = bool>,
>(
    ncw: &T,
) -> Vec<(u32, char, bool, u32)> {
    let mut transitions: Vec<_> = ncw
        .state_indices()
        .flat_map(|q| ncw.transitions_from(q).collect::<Vec<_>>())
        .collect();
    transitions.sort();
    transitions
}
