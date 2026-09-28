//! Examples from \[RK22\] and helpers shared by the tests of the co-Büchi modules.
use super::DCW;
use crate::TransitionSystem;
use automata_core::alphabet::CharAlphabet;

/// RK22, Figure 2: a nice tDCW whose states are all equivalent, but differ in their safe
/// languages. Its safe components are `{q0, q1}` and `{q2}`.
pub(super) const RK22_FIGURE_2: [(u32, char, bool, u32); 9] = [
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

pub(super) fn rk22_figure_2() -> DCW {
    DCW::builder().with_edges(RK22_FIGURE_2).into_dcw(0)
}

/// The tNCW `B_S` for `S = {{q0, q1}}` of RK22, Figure 4, obtained from Figure 2.
pub(super) const RK22_FIGURE_4: [(u32, char, bool, u32); 9] = [
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
