//! Safe language containment, frontiers and safe centralization \[RK22, Section 3.2\].
use super::CoBuchiCondition;
use crate::core::{
    Color,
    alphabet::{Alphabet, Expression},
};
use crate::ts::{Shrinkable, Sproutable};
use crate::{Automaton, TransitionSystem};
use std::collections::{BTreeMap, BTreeSet};

impl<A, Q, D> Automaton<A, CoBuchiCondition, Q, bool, D, true, true>
where
    A: Alphabet,
    Q: Color,
    D: TransitionSystem<Alphabet = A, StateColor = Q, EdgeColor = bool>,
{
    /// Computes the safe language containment relation between the states of `self`, which is
    /// assumed to be safe deterministic. The result contains the pair `(q, s)` iff
    /// `L_safe(A^q) ⊆ L_safe(A^s)`. Following \[RK22\], the safe language `L_safe(A^q)` is the set
    /// of infinite words that can be read from `q` without traversing an `α`-transition.
    ///
    /// Since `self` is safe deterministic, every state has at most one safe run on every word,
    /// so this reduces to containment between deterministic safety automata \[RK22, Section
    /// 3.2\]. First, the states with a nonempty safe language are computed as a greatest
    /// fixpoint: these are the states with a safe transition to such a state. Then,
    /// `L_safe(A^q) ⊈ L_safe(A^s)` iff there is a finite word `u` and a letter `σ` such that the
    /// safe runs of `q` and `s` on `u` exist and end in states `p` and `r`, respectively, `p` has
    /// a safe `σ`-transition to a state with a nonempty safe language, and `r` has no safe
    /// `σ`-transition. The pairs for which this holds are computed as a least fixpoint,
    /// simultaneously for all pairs of states.
    pub fn safe_containment_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        debug_assert!(self.is_safe_deterministic());
        let states: Vec<D::StateIndex> = self.state_indices().collect();

        // safe_successors[q][σ] is the unique `σ`-successor of `q` via a safe transition
        let safe_successors: BTreeMap<D::StateIndex, BTreeMap<A::Symbol, D::StateIndex>> = states
            .iter()
            .map(|&q| {
                let successors = self
                    .transitions_from(q)
                    .filter(|&(_, _, color, _)| !color)
                    .map(|(_, sym, _, target)| (sym, target))
                    .collect();
                (q, successors)
            })
            .collect();

        // states with a nonempty safe language, i.e. with an infinite safe run
        let mut live: BTreeSet<D::StateIndex> = states.iter().copied().collect();
        loop {
            let still_live: BTreeSet<D::StateIndex> = live
                .iter()
                .copied()
                .filter(|q| safe_successors[q].values().any(|p| live.contains(p)))
                .collect();
            if still_live.len() == live.len() {
                break;
            }
            live = still_live;
        }

        // pairs `(q, s)` with `L_safe(A^q) ⊈ L_safe(A^s)`
        let mut not_contained = BTreeSet::new();
        loop {
            let mut changed = false;
            for &q in &states {
                for &s in &states {
                    if not_contained.contains(&(q, s)) {
                        continue;
                    }
                    let witnessed = safe_successors[&q]
                        .iter()
                        .filter(|(_, p)| live.contains(p))
                        .any(|(sym, &p)| match safe_successors[&s].get(sym) {
                            None => true,
                            Some(&r) => not_contained.contains(&(p, r)),
                        });
                    if witnessed {
                        not_contained.insert((q, s));
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }

        states
            .iter()
            .flat_map(|&q| states.iter().map(move |&s| (q, s)))
            .filter(|pair| !not_contained.contains(pair))
            .collect()
    }

    /// Computes the subsafe-equivalence relation between the states of `self`, which is assumed
    /// to be a nice GFG-tNCW. Following \[RK22\], `q` is subsafe-equivalent to `s`, denoted
    /// `q ≤ s`, if `L(A^q) = L(A^s)` and `L_safe(A^q) ⊆ L_safe(A^s)`. The result contains the pair
    /// `(q, s)` iff `q ≤ s`.
    ///
    /// Language equivalence is decided with [`Self::gfg_containment_relation`] and safe language
    /// containment with [`Self::safe_containment_relation`], so the same assumptions apply.
    pub fn subsafe_equivalence_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        let contained = self.gfg_containment_relation();
        self.safe_containment_relation()
            .into_iter()
            .filter(|&(q, s)| contained.contains(&(q, s)) && contained.contains(&(s, q)))
            .collect()
    }

    /// Returns the states of a frontier of `self`, which is assumed to be a nice GFG-tNCW, given
    /// its subsafe-equivalence relation `subsafe` (see [`Self::subsafe_equivalence_relation`]).
    ///
    /// Following \[RK22, Section 3.2\], the relation `H` on the safe components of `self` contains
    /// `(S, S')` iff there are states `q ∈ S` and `q' ∈ S'` with `q ≤ q'`. A frontier is a set of
    /// safe components such that every safe component `S` has some `S'` in the frontier with
    /// `H(S, S')`, and no two different components in the frontier are related by `H`. Since `H`
    /// is reflexive and transitive \[RK22, Lemma 3.9\], a frontier is obtained by taking one
    /// component from each ergodic SCC of the graph induced by `H`, and a component `S` lies in
    /// such an SCC iff `H(S, S')` implies `H(S', S)` for all `S'`. Among the components of an
    /// ergodic SCC, the one containing the smallest state is taken, so the result does not
    /// depend on the order in which the safe components are computed.
    fn frontier_states(
        &self,
        subsafe: &BTreeSet<(D::StateIndex, D::StateIndex)>,
    ) -> BTreeSet<D::StateIndex> {
        let component = self.safe_component_indices();
        let h: BTreeSet<(usize, usize)> = subsafe
            .iter()
            .map(|(q, s)| (component[q], component[s]))
            .collect();

        // the smallest state of every safe component, used to pick a representative
        let mut smallest: BTreeMap<usize, D::StateIndex> = BTreeMap::new();
        for (&q, &c) in &component {
            smallest
                .entry(c)
                .and_modify(|s| *s = (*s).min(q))
                .or_insert(q);
        }

        let in_frontier = |c: usize| {
            let related = || h.iter().filter(move |&&(s, _)| s == c).map(|&(_, t)| t);
            // `c` lies in an ergodic SCC of `H` ...
            related().all(|t| h.contains(&(t, c)))
                // ... and is its representative
                && related().all(|t| smallest[&c] <= smallest[&t])
        };
        component
            .into_iter()
            .filter(|&(_, c)| in_frontier(c))
            .map(|(q, _)| q)
            .collect()
    }

    /// Safe-centralizes `self`, which is assumed to be a nice GFG-tNCW, by turning it into the
    /// tNCW `B_S` of \[RK22, Section 3.2\] for a frontier `S`, in which the smallest state of
    /// each ergodic SCC of `H` determines the component that is taken (see
    /// [`Self::subsafe_equivalence_relation`] for the relation `≤` that `H` is based on).
    /// Afterwards, `self` is a nice, safe-centralized and
    /// `α`-homogenous GFG-tNCW equivalent to the original one \[RK22, Theorem 3.15\].
    ///
    /// `B_S` keeps exactly the states in the safe components of `S`, and for each such state `q`
    /// and letter `σ`:
    /// - if `q` has a safe `σ`-transition, it is kept and all `σ`-labeled `α`-transitions of `q`
    ///   are removed,
    /// - otherwise, `q` gets an `σ`-labeled `α`-transition to every kept state that is
    ///   equivalent to some `σ`-successor of `q`.
    ///
    /// Hence, for every `q` and `σ`, the `σ`-transitions of `q` are either all safe or all
    /// `α`-transitions, i.e. `B_S` is `α`-homogenous. As `self` is normal, safe transitions stay
    /// within their safe component, so no safe transition leads to a removed state. The initial
    /// state is kept if it is in `S`, and is otherwise replaced by the smallest kept state `q'`
    /// with `q_0 ≤ q'`, which exists by \[RK22, Lemma 3.8\]. The indices of all kept states are
    /// unchanged.
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]), or if `B_S` is
    /// nondeterministic but the transition system backing `self` is deterministic, as is the case
    /// for a [`DCW`](super::DCW). Note that even for a deterministic `self`, `B_S` may be
    /// nondeterministic.
    pub fn safe_centralize(&mut self)
    where
        D: Shrinkable + Sproutable,
    {
        assert!(
            self.is_safe_deterministic(),
            "safe centralization requires a safe deterministic tNCW"
        );
        let contained = self.gfg_containment_relation();
        let equivalent = |q, s| contained.contains(&(q, s)) && contained.contains(&(s, q));
        // as in `Self::subsafe_equivalence_relation`, but reusing `contained`
        let subsafe: BTreeSet<_> = self
            .safe_containment_relation()
            .into_iter()
            .filter(|&(q, s)| equivalent(q, s))
            .collect();
        let kept = self.frontier_states(&subsafe);

        if !kept.contains(&self.initial) {
            let initial = self.initial;
            self.initial = kept
                .iter()
                .copied()
                .find(|&q| subsafe.contains(&(initial, q)))
                .expect(
                    "the initial state is subsafe-equivalent to a frontier state [RK22, Lemma 3.8]",
                );
        }

        for &q in &kept {
            let edges = self
                .remove_edges_from(q)
                .expect("state was obtained from state_indices");
            let safe_symbols: BTreeSet<A::Symbol> = edges
                .iter()
                .filter(|&(_, _, color, _)| !color)
                .flat_map(|(_, expression, _, _)| expression.symbols())
                .collect();

            let mut centralized = BTreeSet::new();
            for (source, expression, color, target) in edges {
                if !color {
                    debug_assert!(
                        kept.contains(&target),
                        "safe transition leaves its component"
                    );
                    centralized.insert((source, expression, color, target));
                    continue;
                }
                for sym in expression
                    .symbols()
                    .filter(|sym| !safe_symbols.contains(sym))
                {
                    for &p in kept.iter().filter(|&&p| equivalent(target, p)) {
                        let expression = self.alphabet().make_expression(sym);
                        centralized.insert((source, expression, true, p));
                    }
                }
            }

            let expected = centralized.len();
            for edge in centralized {
                self.add_edge(edge);
            }
            assert_eq!(
                self.edges_from(q).expect("state is kept").count(),
                expected,
                "safe centralization introduces nondeterminism, which the deterministic transition \
                 system backing this automaton cannot represent"
            );
        }

        let removed: Vec<D::StateIndex> =
            self.state_indices().filter(|q| !kept.contains(q)).collect();
        for q in removed {
            self.remove_state(q);
        }
    }

    /// Returns `true` iff `self`, which is assumed to be a nice GFG-tNCW, is safe-centralized.
    /// Following \[RK22\], this is the case if for all states `q` and `s` with `q ≤ s` (see
    /// [`Self::subsafe_equivalence_relation`]), `q` and `s` are in the same safe component.
    /// After [`Self::safe_centralize`], this holds.
    pub fn is_safe_centralized(&self) -> bool {
        let component = self.safe_component_indices();
        self.subsafe_equivalence_relation()
            .into_iter()
            .all(|(q, s)| component[&q] == component[&s])
    }

    /// Returns `true` iff `self` is `α`-homogenous. Following \[RK22\], this is the case if for
    /// every state `q` and symbol `σ`, the `σ`-transitions leaving `q` are either all safe
    /// (colored `false`) or all `α`-transitions (colored `true`). After
    /// [`Self::safe_centralize`], this holds.
    pub fn is_alpha_homogeneous(&self) -> bool {
        self.state_indices().all(|q| {
            let mut colors: BTreeMap<A::Symbol, bool> = BTreeMap::new();
            self.transitions_from(q)
                .all(|(_, sym, color, _)| *colors.entry(sym).or_insert(color) == color)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::rk22_examples::{
        RK22_FIGURE_2, RK22_FIGURE_4, rk22_figure_2, sorted_transitions,
    };
    use super::super::{DCW, NCW};
    use crate::{Pointed, TransitionSystem};
    use automata_core::upw;

    #[test]
    fn rk22_figure_2_safe_containment() {
        // RK22, Example 3.1: the safe languages are `L_safe(q0) = (a + bc)^ω`,
        // `L_safe(q1) = c · L_safe(q0)` and `L_safe(q2) = a^ω`, so apart from the reflexive
        // pairs, only `L_safe(q2) ⊆ L_safe(q0)` holds
        let dcw = DCW::builder()
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
        let contained = dcw.safe_containment_relation();
        assert_eq!(
            contained.into_iter().collect::<Vec<_>>(),
            vec![(0, 0), (1, 1), (2, 0), (2, 2)]
        );
    }

    #[test]
    fn safe_containment_with_empty_safe_languages() {
        // state 0 has a safe transition, but only into state 1, which has no safe transitions
        // at all, so both have an empty safe language and are contained in every state. State 2
        // has the safe language `a^ω`, which is not contained in the empty ones.
        let ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (1, 'a', true, 1),
                (1, 'a', true, 2),
                (2, 'a', false, 2),
            ])
            .into_ncw(0);
        let contained = ncw.safe_containment_relation();
        for s in 0..3 {
            assert!(contained.contains(&(0, s)));
            assert!(contained.contains(&(1, s)));
        }
        assert!(contained.contains(&(2, 2)));
        assert!(!contained.contains(&(2, 0)));
        assert!(!contained.contains(&(2, 1)));
    }

    #[test]
    fn safe_containment_follows_runs_beyond_first_letter() {
        // both 0 and 2 can read `a` safely, but after that, 0 can read `b` safely forever while
        // 2 can only read `c`: the difference is only witnessed after the first letter
        let ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (1, 'b', false, 1),
                (1, 'c', false, 1),
                (1, 'a', true, 0),
                (2, 'a', false, 3),
                (3, 'c', false, 3),
                (3, 'a', true, 2),
                (3, 'b', true, 2),
            ])
            .into_ncw(0);
        let contained = ncw.safe_containment_relation();
        assert!(contained.contains(&(2, 0)));
        assert!(!contained.contains(&(0, 2)));
        assert!(contained.contains(&(3, 1)));
        assert!(!contained.contains(&(1, 3)));
    }

    #[test]
    fn rk22_figure_2_frontier() {
        // RK22, Example 3.10: as all states are equivalent and `L_safe(q2) ⊆ L_safe(q0)`, we have
        // `q2 ≤ q0`, so `H({q2}, {q0, q1})` and the single frontier is `{{q0, q1}}`
        let dcw = rk22_figure_2();
        let subsafe = dcw.subsafe_equivalence_relation();
        assert_eq!(
            subsafe.iter().copied().collect::<Vec<_>>(),
            vec![(0, 0), (1, 1), (2, 0), (2, 2)]
        );
        assert_eq!(
            dcw.frontier_states(&subsafe)
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }

    #[test]
    fn rk22_figure_1_frontier_picks_one_of_strongly_equivalent_components() {
        // RK22, Figure 1: the tDCW `A_fm` for "finitely many `b`s". Both states are
        // strongly-equivalent (safe language `a^ω`), but lie in different safe components
        // `{q0}` and `{q1}`, which are hence related by `H` in both directions. Only one of them
        // may be in the frontier, and the one with the smaller state is taken.
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(0);
        let subsafe = dcw.subsafe_equivalence_relation();
        assert_eq!(subsafe.len(), 4);
        assert_eq!(
            dcw.frontier_states(&subsafe)
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0]
        );
    }

    #[test]
    fn frontier_keeps_components_with_different_languages() {
        // L(q1) is "finitely many `a`s", and L(q0) = a^ω + a^* b L(q1). The states are not
        // equivalent, so `H` only relates each safe component to itself, and both are kept.
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
            ])
            .into_dcw(0);
        assert!(dcw.is_normal());
        let subsafe = dcw.subsafe_equivalence_relation();
        assert_eq!(
            subsafe.iter().copied().collect::<Vec<_>>(),
            vec![(0, 0), (1, 1)]
        );
        assert_eq!(
            dcw.frontier_states(&subsafe)
                .into_iter()
                .collect::<Vec<_>>(),
            vec![0, 1]
        );

        // hence, safe centralization changes nothing
        let mut centralized = dcw.clone();
        centralized.safe_centralize();
        assert_eq!(sorted_transitions(&centralized), sorted_transitions(&dcw));
    }

    #[test]
    fn rk22_figure_2_safe_centralizes_to_figure_4() {
        // RK22, Example 3.10: the state q2 is removed, and every `α`-transition of q0 and q1
        // that is not overruled by a safe transition on the same letter is redirected to both
        // q0 and q1, as all states are equivalent
        let mut ncw = NCW::builder().with_edges(RK22_FIGURE_2).into_ncw(0);
        ncw.safe_centralize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(sorted_transitions(&ncw), RK22_FIGURE_4.to_vec());
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.is_normal());
    }

    #[test]
    fn safe_centralize_moves_initial_state_out_of_removed_component() {
        // same as `rk22_figure_2_safe_centralizes_to_figure_4`, but starting in q2, which is
        // removed. As `q2 ≤ q0` but not `q2 ≤ q1`, the new initial state is q0.
        let mut ncw = NCW::builder().with_edges(RK22_FIGURE_2).into_ncw(2);
        ncw.safe_centralize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), RK22_FIGURE_4.to_vec());
    }

    #[test]
    fn rk22_figure_1_safe_centralizes_to_single_state() {
        // RK22, Figure 1: only the safe component {q0} is kept, and the `α`-transition on `b` is
        // redirected to q0 itself. The result is deterministic, so a `DCW` can represent it.
        let mut dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(1);
        let words = [upw!("a"), upw!("b"), upw!("ba", "a"), upw!("a", "ab")];
        let accepted: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();

        dcw.safe_centralize();
        assert_eq!(dcw.initial(), 0);
        assert_eq!(
            sorted_transitions(&dcw),
            vec![(0, 'a', false, 0), (0, 'b', true, 0)]
        );
        let accepted_after: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();
        assert_eq!(accepted, accepted_after);
        assert_eq!(accepted, vec![true, false, true, false]);
    }

    #[test]
    fn rk22_figure_2_as_dcw_safe_centralizes_after_into_ncw() {
        // unlike in `safe_centralize_panics_if_dcw_cannot_represent_result`, converting the `DCW`
        // first allows the nondeterministic `α`-transitions of Figure 4
        let mut ncw = rk22_figure_2().into_ncw();
        ncw.safe_centralize();
        assert_eq!(sorted_transitions(&ncw), RK22_FIGURE_4.to_vec());
    }

    #[test]
    #[should_panic(expected = "deterministic transition system")]
    fn safe_centralize_panics_if_dcw_cannot_represent_result() {
        // RK22, Figure 2 as a `DCW`: `B_S` has two `c`-transitions from q0, which the
        // deterministic backing transition system cannot represent
        let mut dcw = rk22_figure_2();
        dcw.safe_centralize();
    }

    #[test]
    #[should_panic(expected = "safe deterministic")]
    fn safe_centralize_panics_if_not_safe_deterministic() {
        // same automaton as in `ncw_not_safe_deterministic_with_two_safe_transitions`
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', false, 2),
                (1, 'a', false, 1),
                (2, 'a', false, 2),
            ])
            .into_ncw(0);
        ncw.safe_centralize();
    }

    #[test]
    fn safe_centralization_checks_on_rk22_figures() {
        // RK22, Example 3.1: Figure 2 is not safe-centralized, as `q2 ≤ q0` but they are in
        // different safe components. Being deterministic, it is trivially `α`-homogenous.
        let mut ncw = rk22_figure_2().into_ncw();
        assert!(!ncw.is_safe_centralized());
        assert!(ncw.is_alpha_homogeneous());
        ncw.safe_centralize();
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_alpha_homogeneous());

        // RK22, Example 3.1: Figure 1 is not safe-centralized, as its two states are
        // strongly-equivalent but in different safe components
        let mut dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(0);
        assert!(!dcw.is_safe_centralized());
        dcw.safe_centralize();
        assert!(dcw.is_safe_centralized());
        assert!(dcw.is_alpha_homogeneous());
    }

    #[test]
    fn not_alpha_homogeneous_with_safe_and_alpha_transition_on_same_letter() {
        // same automaton as in `ncw_safe_deterministic_despite_nondeterminism_on_alpha_transitions`:
        // state 0 has both a safe and an `α`-transition on `a`
        let ncw = NCW::builder()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_ncw(0);
        assert!(!ncw.is_alpha_homogeneous());

        // several `α`-transitions on the same letter are fine
        let ncw = NCW::builder().with_edges(RK22_FIGURE_4).into_ncw(0);
        assert!(ncw.is_alpha_homogeneous());
    }
}
