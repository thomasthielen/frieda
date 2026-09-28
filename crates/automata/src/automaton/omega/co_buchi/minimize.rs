//! Strong equivalence and safe minimization \[RK22, Section 3.3\].
use super::CoBuchiCondition;
use crate::core::{Color, alphabet::Alphabet};
use crate::ts::{Shrinkable, Sproutable};
use crate::{Automaton, TransitionSystem};
use std::collections::{BTreeMap, BTreeSet};

impl<A, Q, D> Automaton<A, CoBuchiCondition, Q, bool, D, true, true>
where
    A: Alphabet,
    Q: Color,
    D: TransitionSystem<Alphabet = A, StateColor = Q, EdgeColor = bool>,
{
    /// Computes the strong-equivalence relation between the states of `self`, which is assumed
    /// to be a nice GFG-tNCW. Following \[RK22\], `q` and `s` are strongly-equivalent, denoted
    /// `q ≈ s`, if `L(A^q) = L(A^s)` and `L_safe(A^q) = L_safe(A^s)`. The result contains the pair
    /// `(q, s)` iff `q ≈ s`.
    ///
    /// As `q ≈ s` iff `q ≤ s` and `s ≤ q`, this is derived from
    /// [`Self::subsafe_equivalence_relation`], so the same assumptions apply.
    pub fn strong_equivalence_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        let subsafe = self.subsafe_equivalence_relation();
        subsafe
            .iter()
            .copied()
            .filter(|&(q, s)| subsafe.contains(&(s, q)))
            .collect()
    }

    /// Returns `true` iff `self`, which is assumed to be a nice GFG-tNCW, is safe-minimal.
    /// Following \[RK22\], this is the case if no two different states are strongly-equivalent
    /// (see [`Self::strong_equivalence_relation`]). After [`Self::safe_minimize`], this holds.
    pub fn is_safe_minimal(&self) -> bool {
        self.strong_equivalence_relation()
            .into_iter()
            .all(|(q, s)| q == s)
    }

    /// Safe-minimizes `self`, which is assumed to be a nice, safe-centralized and `α`-homogenous
    /// GFG-tNCW (as obtained by [`Self::safe_centralize`]), by turning it into the quotient tNCW
    /// `C` of \[RK22, Section 3.3\], in which strongly-equivalent states are merged (see
    /// [`Self::strong_equivalence_relation`]). Afterwards, `self` is a nice, safe-centralized,
    /// safe-minimal and `α`-homogenous GFG-tNCW equivalent to the original one \[RK22, Theorem
    /// 3.20\], and hence a minimal GFG-tNCW \[RK22, Theorem 3.6\].
    ///
    /// Each equivalence class `[q]` of `≈` is represented by its smallest state, which is kept,
    /// while all other states are removed. The indices of the kept states are unchanged. There
    /// is a transition `⟨[q], σ, [p]⟩` iff there are `q' ∈ [q]` and `p' ∈ [p]` with a transition
    /// `⟨q', σ, p'⟩`, and it is an `α`-transition iff `⟨q', σ, p'⟩` is. Since `self` is
    /// `α`-homogenous and strongly-equivalent states have the same safe language, this does not
    /// depend on the choice of `q'` and `p'`. The initial state is replaced by the representative
    /// of its class.
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]) or not
    /// `α`-homogenous (see [`Self::is_alpha_homogeneous`]), or if `C` is nondeterministic but the
    /// transition system backing `self` is deterministic, as is the case for a
    /// [`DCW`](super::DCW). Note that even for a deterministic `self`, `C` may be
    /// nondeterministic, as the `α`-transitions of the merged states are joined.
    pub fn safe_minimize(&mut self)
    where
        D: Shrinkable + Sproutable,
    {
        assert!(
            self.is_safe_deterministic(),
            "safe minimization requires a safe deterministic tNCW"
        );
        assert!(
            self.is_alpha_homogeneous(),
            "safe minimization requires an α-homogenous tNCW"
        );

        // the representative of `[q]` is its smallest state, and `(q, s)` with `s ≤ q` is
        // visited first, as the pairs are ordered lexicographically
        let mut representative: BTreeMap<D::StateIndex, D::StateIndex> = BTreeMap::new();
        for (q, s) in self.strong_equivalence_relation() {
            representative.entry(q).or_insert(s);
        }
        self.initial = representative[&self.initial];

        let states: Vec<D::StateIndex> = self.state_indices().collect();
        let mut quotient: BTreeMap<D::StateIndex, BTreeSet<_>> = BTreeMap::new();
        for q in states {
            let edges = self
                .remove_edges_from(q)
                .expect("state was obtained from state_indices");
            let source = representative[&q];
            let merged = quotient.entry(source).or_default();
            for (_, expression, color, target) in edges {
                merged.insert((source, expression, color, representative[&target]));
            }
        }

        let removed: Vec<D::StateIndex> = representative
            .iter()
            .filter(|(q, s)| q != s)
            .map(|(&q, _)| q)
            .collect();
        for q in removed {
            self.remove_state(q);
        }

        for (q, edges) in quotient {
            let expected = edges.len();
            for edge in edges {
                self.add_edge(edge);
            }
            assert_eq!(
                self.edges_from(q).expect("state is kept").count(),
                expected,
                "safe minimization introduces nondeterminism, which the deterministic transition \
                 system backing this automaton cannot represent"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::rk22_examples::{RK22_FIGURE_4, rk22_figure_2, sorted_transitions};
    use super::super::{DCW, NCW};
    use crate::{Pointed, TransitionSystem};
    use automata_core::upw;

    /// RK22, Figure 4 with q0 split into the strongly-equivalent states 0 and 2, which alternate
    /// on `a`. All states are equivalent, as in Figure 4, and the automaton is deterministic,
    /// nice, safe-centralized (it has a single safe component) and `α`-homogenous. The
    /// `α`-transitions on `c` of 0 and 2 lead to different states, so merging 0 and 2 yields a
    /// nondeterministic `c`-transition.
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
    fn rk22_figure_1_strong_equivalence() {
        // RK22, Example 3.1: the two states of `A_fm` are strongly-equivalent, so it is not
        // safe-minimal
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(0);
        assert_eq!(dcw.strong_equivalence_relation().len(), 4);
        assert!(!dcw.is_safe_minimal());
    }

    #[test]
    fn rk22_figure_2_is_safe_minimal() {
        // RK22, Example 3.1: all states of Figure 2 differ in their safe languages, even though
        // `q2 ≤ q0`
        let dcw = rk22_figure_2();
        assert_eq!(
            dcw.strong_equivalence_relation()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![(0, 0), (1, 1), (2, 2)]
        );
        assert!(dcw.is_safe_minimal());
    }

    #[test]
    fn rk22_figure_4_safe_minimizes_to_itself() {
        // RK22, Example 3.21: q0 and q1 differ in their safe languages, so `C` is identical to
        // `B_S`
        let mut ncw = NCW::builder().with_edges(RK22_FIGURE_4).into_ncw(0);
        assert!(ncw.is_safe_minimal());
        ncw.safe_minimize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), RK22_FIGURE_4.to_vec());
    }

    #[test]
    fn safe_minimize_merges_strongly_equivalent_states_in_one_safe_component() {
        // unlike in RK22, Figure 1, the strongly-equivalent states 0 and 1 are in the same safe
        // component, so safe centralization keeps both and safe minimization merges them. The
        // result is deterministic, so a `DCW` can represent it.
        let mut dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'b', true, 0),
                (1, 'a', false, 0),
                (1, 'b', true, 1),
            ])
            .into_dcw(1);
        assert!(dcw.is_safe_centralized());
        assert!(!dcw.is_safe_minimal());
        let words = [upw!("a"), upw!("b"), upw!("ba", "a"), upw!("a", "ab")];
        let accepted: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();

        dcw.safe_minimize();
        assert_eq!(dcw.initial(), 0);
        assert_eq!(dcw.state_indices().collect::<Vec<_>>(), vec![0]);
        assert_eq!(
            sorted_transitions(&dcw),
            vec![(0, 'a', false, 0), (0, 'b', true, 0)]
        );
        let accepted_after: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();
        assert_eq!(accepted, accepted_after);
        assert_eq!(accepted, vec![true, false, true, false]);
        assert!(dcw.is_safe_minimal());
    }

    #[test]
    fn safe_minimize_joins_alpha_transitions_of_merged_states() {
        let mut ncw = NCW::builder().with_edges(SPLIT_FIGURE_4).into_ncw(2);
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_alpha_homogeneous());
        assert_eq!(
            ncw.strong_equivalence_relation()
                .into_iter()
                .collect::<Vec<_>>(),
            vec![(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)]
        );
        assert!(!ncw.is_safe_minimal());

        ncw.safe_minimize();
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
        // the result is nice, safe-centralized, safe-minimal and `α`-homogenous
        // [RK22, Proposition 3.19]
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.remove_unreachable_states().is_empty());
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_safe_minimal());
        assert!(ncw.is_alpha_homogeneous());
    }

    #[test]
    #[should_panic(expected = "deterministic transition system")]
    fn safe_minimize_panics_if_dcw_cannot_represent_result() {
        let mut dcw = DCW::builder().with_edges(SPLIT_FIGURE_4).into_dcw(0);
        dcw.safe_minimize();
    }

    #[test]
    #[should_panic(expected = "α-homogenous")]
    fn safe_minimize_panics_if_not_alpha_homogeneous() {
        // same automaton as in `not_alpha_homogeneous_with_safe_and_alpha_transition_on_same_letter`
        let mut ncw = NCW::builder()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_ncw(0);
        ncw.safe_minimize();
    }
}
