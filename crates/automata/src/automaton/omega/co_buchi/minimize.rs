//! Strong equivalence and safe minimization \[AK22, Section 3.3\].
//!
//! \[AK22\] Bader Abu Radi, Orna Kupferman: *Minimization and Canonization of GFG
//! Transition-Based Automata*. Log. Methods Comput. Sci. 18(3) (2022)
use super::{CoBuchiCondition, NCW};
use crate::core::{Color, alphabet::Alphabet};
use crate::ts::{Shrinkable, StateIndex};
use crate::{Automaton, TransitionSystem};
use std::collections::{BTreeMap, BTreeSet};

/// Returns the strong-equivalence relation `≈` given the subsafe-equivalence relation `≤`, as
/// `q ≈ s` iff `q ≤ s` and `s ≤ q`.
fn strong_equivalence<S: Ord + Copy>(subsafe: &BTreeSet<(S, S)>) -> BTreeSet<(S, S)> {
    subsafe
        .iter()
        .copied()
        .filter(|&(q, s)| subsafe.contains(&(s, q)))
        .collect()
}

impl<A, Q, D> Automaton<A, CoBuchiCondition, Q, bool, D, true, true>
where
    A: Alphabet,
    Q: Color,
    D: TransitionSystem<Alphabet = A, StateColor = Q, EdgeColor = bool>,
{
    /// Computes the strong-equivalence relation between the states of `self`, which is assumed
    /// to be a nice GFG-tNCW. Following \[AK22\], `q` and `s` are strongly-equivalent, denoted
    /// `q ≈ s`, if `L(A^q) = L(A^s)` and `L_safe(A^q) = L_safe(A^s)`. The result contains the pair
    /// `(q, s)` iff `q ≈ s`.
    ///
    /// As `q ≈ s` iff `q ≤ s` and `s ≤ q`, this is derived from
    /// [`Self::subsafe_equivalence_relation`], so the same assumptions apply.
    pub fn strong_equivalence_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        strong_equivalence(&self.subsafe_equivalence_relation())
    }

    /// Returns `true` iff `self`, which is assumed to be a nice GFG-tNCW, is safe-minimal.
    /// Following \[AK22\], this is the case if no two different states are strongly-equivalent
    /// (see [`Self::strong_equivalence_relation`]). After [`Self::safe_minimize`], this holds.
    pub fn is_safe_minimal(&self) -> bool {
        self.strong_equivalence_relation()
            .into_iter()
            .all(|(q, s)| q == s)
    }
}

impl<A: Alphabet, Q: Color> NCW<A, Q> {
    /// Safe-minimizes `self`, which is assumed to be a nice, safe-centralized and `α`-homogenous
    /// GFG-tNCW (as obtained by [`Self::safe_centralize`]), by turning it into the quotient tNCW
    /// `C` of \[AK22, Section 3.3\], in which strongly-equivalent states are merged (see
    /// [`Self::strong_equivalence_relation`]). Afterwards, `self` is a nice, safe-centralized,
    /// safe-minimal and `α`-homogenous GFG-tNCW equivalent to the original one \[AK22, Theorem
    /// 3.20\], and hence a minimal GFG-tNCW \[AK22, Theorem 3.6\].
    ///
    /// Each equivalence class `[q]` of `≈` is represented by its smallest state, which is kept,
    /// while all other states are removed. The indices of the kept states are unchanged. There
    /// is a transition `⟨[q], σ, [p]⟩` iff there are `q' ∈ [q]` and `p' ∈ [p]` with a transition
    /// `⟨q', σ, p'⟩`, and it is an `α`-transition iff `⟨q', σ, p'⟩` is. Since `self` is
    /// `α`-homogenous and strongly-equivalent states have the same safe language, this does not
    /// depend on the choice of `q'` and `p'`. The initial state is replaced by the representative
    /// of its class.
    ///
    /// As the `α`-transitions of the merged states are joined, `C` may be nondeterministic, even
    /// if `self` is deterministic. Hence, this is only available on an [`NCW`]; use
    /// [`DCW::into_safe_minimized`](super::DCW::into_safe_minimized) to safe-minimize a
    /// [`DCW`](super::DCW).
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]) or not
    /// `α`-homogenous (see [`Self::is_alpha_homogeneous`]).
    pub fn safe_minimize(&mut self) {
        assert!(
            self.is_safe_deterministic(),
            "safe minimization requires a safe deterministic tNCW"
        );
        assert!(
            self.is_alpha_homogeneous(),
            "safe minimization requires an α-homogenous tNCW"
        );
        let subsafe = self.subsafe_equivalence_relation();
        self.safe_minimize_with(&subsafe);
    }

    /// Does the same as [`Self::safe_minimize`], but derives strong equivalence from the given
    /// subsafe-equivalence relation `subsafe` (see [`Self::subsafe_equivalence_relation`]) instead
    /// of computing it, e.g. from the one returned by [`Self::safe_centralize_with`].
    pub(super) fn safe_minimize_with(
        &mut self,
        subsafe: &BTreeSet<(StateIndex<Self>, StateIndex<Self>)>,
    ) {
        debug_assert!(self.is_safe_deterministic());
        debug_assert!(self.is_alpha_homogeneous());

        // the representative of `[q]` is its smallest state, and `(q, s)` with `s ≤ q` is
        // visited first, as the pairs are ordered lexicographically
        let mut representative: BTreeMap<_, _> = BTreeMap::new();
        for (q, s) in strong_equivalence(subsafe) {
            representative.entry(q).or_insert(s);
        }
        self.initial = representative[&self.initial];

        let states: Vec<_> = self.state_indices().collect();
        let mut quotient = vec![];
        for q in states {
            let edges = self
                .remove_edges_from(q)
                .expect("state was obtained from state_indices");
            let source = representative[&q];
            for (_, expression, color, target) in edges {
                quotient.push((source, expression, color, representative[&target]));
            }
        }

        let removed: Vec<_> = representative
            .iter()
            .filter(|(q, s)| q != s)
            .map(|(&q, _)| q)
            .collect();
        for q in removed {
            self.remove_state(q);
        }

        self.add_edges_without_duplicates(quotient);
    }
}

#[cfg(test)]
mod tests {
    use super::super::ak22_examples::{
        AK22_FIGURE_4, ak22_figure_1, ak22_figure_2, prop_ncw, safe_and_alpha_on_same_letter_ncw,
        sorted_transitions,
    };
    use super::super::{DCW, NCW};
    use crate::{Pointed, TransitionSystem};

    /// AK22, Figure 4 with q0 split into the strongly-equivalent states 0 and 2, which alternate
    /// on `a`. It is deterministic, nice, safe-centralized and `α`-homogenous, but the
    /// `α`-transitions on `c` of 0 and 2 lead to different states.
    const SPLIT_FIGURE_4: [(u32, char, bool, u32); 9] = [
        (0, 'a', false, 2),
        (0, 'b', false, 1),
        (0, 'c', true, 0),
        (1, 'a', true, 0),
        (1, 'b', true, 0),
        (1, 'c', false, 0),
        (2, 'a', false, 0),
        (2, 'b', false, 1),
        (2, 'c', true, 1),
    ];

    #[test]
    fn ak22_figure_1_strong_equivalence() {
        // AK22, Example 3.1
        let dcw = ak22_figure_1();
        assert_eq!(dcw.strong_equivalence_relation().len(), 4);
        assert!(!dcw.is_safe_minimal());
    }

    #[test]
    fn ak22_figure_2_is_safe_minimal() {
        // AK22, Example 3.1
        let dcw = ak22_figure_2();
        assert_eq!(
            dcw.strong_equivalence_relation()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 1), (2, 2)]
        );
        assert!(dcw.is_safe_minimal());
    }

    #[test]
    fn ak22_figure_4_safe_minimizes_to_itself() {
        // AK22, Example 3.21
        let mut ncw = NCW::builder().with_edges(AK22_FIGURE_4).into_ncw(0);
        assert!(ncw.is_safe_minimal());
        ncw.safe_minimize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
    }

    #[test]
    fn safe_minimize_merges_strongly_equivalent_states_in_one_safe_component() {
        // unlike in AK22, Figure 1, the strongly-equivalent 0 and 1 share a safe component
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'b', true, 0),
                (1, 'a', false, 0),
                (1, 'b', true, 1),
            ])
            .into_dcw(1);
        assert!(dcw.is_safe_centralized());
        assert!(!dcw.is_safe_minimal());
        let dcw = dcw
            .into_safe_minimized()
            .try_into_dcw()
            .expect("the result is deterministic");
        assert_eq!(dcw.initial(), 0);
        assert_eq!(
            sorted_transitions(&dcw),
            vec![(0, 'a', false, 0), (0, 'b', true, 0)]
        );
        assert!(dcw.is_safe_minimal());
    }

    #[test]
    fn safe_minimize_joins_alpha_transitions_of_merged_states() {
        let dcw = DCW::builder().with_edges(SPLIT_FIGURE_4).into_dcw(2);
        assert!(dcw.is_semantically_deterministic());
        assert!(dcw.is_normal());
        assert!(dcw.is_safe_centralized());
        assert!(dcw.is_alpha_homogeneous());
        assert_eq!(
            dcw.strong_equivalence_relation()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
        );
        assert!(!dcw.is_safe_minimal());

        let mut ncw = dcw.into_safe_minimized();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', false, 0),
                (0, 'b', false, 1),
                (0, 'c', true, 0),
                (0, 'c', true, 1),
                (1, 'a', true, 0),
                (1, 'b', true, 0),
                (1, 'c', false, 0),
            ]
        );
        // AK22, Proposition 3.19
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.remove_unreachable_states().is_empty());
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_safe_minimal());
        assert!(ncw.is_alpha_homogeneous());
        assert!(ncw.try_into_dcw().is_err());
    }

    #[test]
    fn safe_minimize_merges_overlapping_edges_of_merged_states() {
        // the universal states 0 and 1 are strongly-equivalent and are merged into 0. Each reads
        // both symbols safely into the other, 0 with a single edge and 1 with one edge per
        // symbol, so the quotient must not get two safe transitions on the same symbol.
        let mut ncw = prop_ncw(
            2,
            [
                (0, "x_0 | !x_0", false, 1),
                (1, "x_0", false, 0),
                (1, "!x_0", false, 0),
            ],
        );
        assert!(!ncw.is_safe_minimal());
        ncw.safe_minimize();
        assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![0]);
        assert_eq!(ncw.transitions_from(0).count(), 2);
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_safe_minimal());
    }

    #[test]
    #[should_panic(expected = "α-homogenous")]
    fn safe_minimize_panics_if_not_alpha_homogeneous() {
        safe_and_alpha_on_same_letter_ncw().safe_minimize();
    }
}
