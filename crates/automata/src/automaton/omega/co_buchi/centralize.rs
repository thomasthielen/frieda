//! Safe language containment, frontiers and safe centralization \[AK22, Section 3.2\].
//!
//! \[AK22\] Bader Abu Radi, Orna Kupferman: *Minimization and Canonization of GFG
//! Transition-Based Automata*. Log. Methods Comput. Sci. 18(3) (2022)
use super::determinize::LanguageClasses;
use super::{CoBuchiCondition, NCW};
use crate::core::{
    Color,
    alphabet::{Alphabet, Expression},
};
use crate::ts::{Shrinkable, Sproutable, StateIndex};
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
    /// `L_safe(A^q) ⊆ L_safe(A^s)`. Following \[AK22\], the safe language `L_safe(A^q)` is the set
    /// of infinite words that can be read from `q` without traversing an `α`-transition.
    pub fn safe_containment_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        let states: Vec<D::StateIndex> = self.state_indices().collect();
        let pairs = states
            .iter()
            .flat_map(|&q| states.iter().map(move |&s| (q, s)))
            .collect();
        self.safe_containment_among(pairs)
    }

    /// Computes the pairs `(q, s)` among the `candidates` with `L_safe(A^q) ⊆ L_safe(A^s)`, where
    /// `self` is assumed to be safe deterministic, see [`Self::safe_containment_relation`]. Pairs
    /// that are no candidates are treated as not contained, so the result is exact if for every
    /// candidate `(q, s)` and symbol `σ`, the safe `σ`-successors of `q` and `s` (if both exist)
    /// form a candidate again.
    fn safe_containment_among(
        &self,
        candidates: BTreeSet<(D::StateIndex, D::StateIndex)>,
    ) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
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
        // via greatest fixpoint: a state is live iff it has a safe successor that is live
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

        // safe_predecessors[p][σ] are the states with a safe `σ`-transition to `p`
        let mut safe_predecessors: BTreeMap<D::StateIndex, BTreeMap<A::Symbol, Vec<_>>> =
            BTreeMap::new();
        for (&q, successors) in &safe_successors {
            for (&sym, &p) in successors {
                safe_predecessors
                    .entry(p)
                    .or_default()
                    .entry(sym)
                    .or_default()
                    .push(q);
            }
        }

        // pairs `(q, s)` with `L_safe(A^q) ⊆ L_safe(A^s)` - via greatest fixpoint: a pair is
        // refuted if some live safe `σ`-successor `p` of `q` is not matched by a safe
        // `σ`-successor `r` of `s` with `(p, r)` still contained
        let mut contained = candidates;
        let mut refuted: Vec<_> = contained
            .iter()
            .copied()
            .filter(|(q, s)| {
                safe_successors[q]
                    .iter()
                    .filter(|(_, p)| live.contains(p))
                    .any(|(sym, &p)| match safe_successors[s].get(sym) {
                        None => true,
                        Some(&r) => !contained.contains(&(p, r)),
                    })
            })
            .collect();
        for pair in &refuted {
            contained.remove(pair);
        }
        // a refuted pair `(p, r)` (where `p` is live, as it has a live safe successor) refutes
        // every contained `(q, s)` with safe `σ`-transitions `q → p` and `s → r`
        let empty = BTreeMap::new();
        while let Some((p, r)) = refuted.pop() {
            let predecessors_of_r = safe_predecessors.get(&r).unwrap_or(&empty);
            for (sym, qs) in safe_predecessors.get(&p).unwrap_or(&empty) {
                let Some(ss) = predecessors_of_r.get(sym) else {
                    continue;
                };
                for &q in qs {
                    for &s in ss {
                        if contained.remove(&(q, s)) {
                            refuted.push((q, s));
                        }
                    }
                }
            }
        }
        contained
    }

    /// Computes the subsafe-equivalence relation between the states of `self`, which is assumed
    /// to be a nice GFG-tNCW. Following \[AK22\], `q` is subsafe-equivalent to `s`, denoted
    /// `q ≤ s`, if `L(A^q) = L(A^s)` and `L_safe(A^q) ⊆ L_safe(A^s)`. The result contains the pair
    /// `(q, s)` iff `q ≤ s`.
    pub fn subsafe_equivalence_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        let contained = self.gfg_containment_relation();
        let classes = LanguageClasses::from_containment(&contained, self.state_indices());
        self.subsafe_equivalence_relation_with(&classes)
    }

    /// Does the same as [`Self::subsafe_equivalence_relation`], but takes language equivalence
    /// from the given partition `classes` instead of computing it.
    ///
    /// Safe containment is only computed between equivalent states: as `self` is semantically
    /// deterministic, the `σ`-successors of equivalent states are equivalent again, so the safe
    /// containment between equivalent states only depends on that between other equivalent
    /// states.
    pub(super) fn subsafe_equivalence_relation_with(
        &self,
        classes: &LanguageClasses<D::StateIndex>,
    ) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        let states: Vec<D::StateIndex> = self.state_indices().collect();
        let candidates = states
            .iter()
            .flat_map(|&q| states.iter().map(move |&s| (q, s)))
            .filter(|&(q, s)| classes.equivalent(q, s))
            .collect();
        self.safe_containment_among(candidates)
    }

    /// Returns the states of a frontier of `self`, which is assumed to be a nice GFG-tNCW, given
    /// its subsafe-equivalence relation `subsafe` (see [`Self::subsafe_equivalence_relation`]).
    ///
    /// Following \[AK22, Section 3.2\], the relation `H` on the safe components of `self` contains
    /// `(S, S')` iff there are states `q ∈ S` and `q' ∈ S'` with `q ≤ q'`. A frontier is a set of
    /// safe components such that every safe component `S` has some `S'` in the frontier with
    /// `H(S, S')`, and no two different components in the frontier are related by `H`. Since `H`
    /// is reflexive and transitive \[AK22, Lemma 3.9\], a frontier is obtained by taking one
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

        // the smallest state of every safe component, used to pick a representative; the states
        // are visited in ascending order, so the first one seen of a component is its smallest
        let mut smallest: BTreeMap<usize, D::StateIndex> = BTreeMap::new();
        for (&q, &c) in &component {
            smallest.entry(c).or_insert(q);
        }

        // all `S'` with `H(S, S')`
        let related = |c: usize| h.range((c, 0)..=(c, usize::MAX)).map(|&(_, t)| t);
        let frontier: BTreeSet<usize> = smallest
            .keys()
            .copied()
            .filter(|&c| {
                // `c` lies in an ergodic SCC of `H` ...
                related(c).all(|t| h.contains(&(t, c)))
                    // ... and is its representative
                    && related(c).all(|t| smallest[&c] <= smallest[&t])
            })
            .collect();
        component
            .into_iter()
            .filter(|(_, c)| frontier.contains(c))
            .map(|(q, _)| q)
            .collect()
    }

    /// Returns `true` iff `self`, which is assumed to be a nice GFG-tNCW, is safe-centralized.
    /// Following \[AK22\], this is the case if for all states `q` and `s` with `q ≤ s` (see
    /// [`Self::subsafe_equivalence_relation`]), `q` and `s` are in the same safe component.
    /// After [`Self::safe_centralize`], this holds.
    pub fn is_safe_centralized(&self) -> bool {
        let component = self.safe_component_indices();
        self.subsafe_equivalence_relation()
            .into_iter()
            .all(|(q, s)| component[&q] == component[&s])
    }

    /// Returns `true` iff `self` is `α`-homogenous. Following \[AK22\], this is the case if for
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

impl<A: Alphabet, Q: Color> NCW<A, Q> {
    /// Safe-centralizes `self`, which is assumed to be a nice GFG-tNCW, by turning it into the
    /// tNCW `B_S` of \[AK22, Section 3.2\], where the frontier `S` takes the component of the
    /// smallest state of each ergodic SCC of `H` (see [`Self::subsafe_equivalence_relation`]).
    /// Afterwards, `self` is a nice, safe-centralized and `α`-homogenous GFG-tNCW equivalent to
    /// the original one \[AK22, Theorem 3.15\].
    ///
    /// Only the states of `S` are kept, with unchanged indices. For each kept `q` and letter `σ`,
    /// a safe `σ`-transition of `q` is kept and its `σ`-labeled `α`-transitions are removed;
    /// without one, `q` gets a `σ`-labeled `α`-transition to every kept state equivalent to some
    /// `σ`-successor of `q`. If `q_0 ∉ S`, it is replaced by the smallest kept `q'` with
    /// `q_0 ≤ q'` \[AK22, Lemma 3.8\].
    ///
    /// As `B_S` may be nondeterministic, even if `self` is deterministic, this is only available
    /// on an [`NCW`]; use [`DCW::into_safe_centralized`](super::DCW::into_safe_centralized) to
    /// safe-centralize a [`DCW`](super::DCW).
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]).
    pub fn safe_centralize(&mut self) {
        assert!(
            self.is_safe_deterministic(),
            "safe centralization requires a safe deterministic tNCW"
        );
        let contained = self.gfg_containment_relation();
        let classes = LanguageClasses::from_containment(&contained, self.state_indices());
        self.safe_centralize_with(&classes);
    }

    /// Does the same as [`Self::safe_centralize`], but takes language equivalence from the given
    /// partition `classes` instead of computing it. Returns the subsafe-equivalence relation `≤`
    /// of the result (see [`Self::subsafe_equivalence_relation`]), so that
    /// [`Self::safe_minimize_with`] can reuse it.
    ///
    /// This is the subsafe-equivalence relation of `self` before safe centralization, restricted
    /// to the kept states: their languages are unchanged \[AK22, Theorem 3.15\], and so are their
    /// safe languages, as all their safe transitions are kept and stay within their safe
    /// component, which is kept as a whole.
    pub(super) fn safe_centralize_with(
        &mut self,
        classes: &LanguageClasses<StateIndex<Self>>,
    ) -> BTreeSet<(StateIndex<Self>, StateIndex<Self>)> {
        debug_assert!(self.is_safe_deterministic());
        // q ≤ s
        let mut subsafe = self.subsafe_equivalence_relation_with(classes);
        let kept = self.frontier_states(&subsafe);

        if !kept.contains(&self.initial) {
            let initial = self.initial;
            self.initial = kept
                .iter()
                .copied()
                .find(|&q| subsafe.contains(&(initial, q)))
                .expect(
                    "the initial state is subsafe-equivalent to a frontier state [AK22, Lemma 3.8]",
                );
        }

        // the kept states of every language class, each of which has one [AK22, Lemma 3.8]
        let mut kept_in_class: BTreeMap<_, Vec<_>> = BTreeMap::new();
        for &p in &kept {
            kept_in_class.entry(classes.class(p)).or_default().push(p);
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
                    for &p in &kept_in_class[&classes.class(target)] {
                        let expression = self.alphabet().make_expression(sym);
                        centralized.insert((source, expression, true, p));
                    }
                }
            }

            for edge in centralized {
                self.add_edge(edge);
            }
        }

        let removed: Vec<_> = self.state_indices().filter(|q| !kept.contains(q)).collect();
        for q in removed {
            self.remove_state(q);
        }

        subsafe.retain(|(q, s)| kept.contains(q) && kept.contains(s));
        subsafe
    }
}

#[cfg(test)]
mod tests {
    use super::super::ak22_examples::{
        AK22_FIGURE_1, AK22_FIGURE_2, AK22_FIGURE_4, ak22_figure_1, ak22_figure_2,
        not_safe_deterministic_ncw, safe_and_alpha_on_same_letter_ncw, sorted_transitions,
    };
    use super::super::{DCW, NCW};
    use crate::Pointed;

    #[test]
    fn safe_containment() {
        // AK22, Example 3.1: `L_safe(q0) = (a + bc)^ω`, `L_safe(q1) = c · L_safe(q0)` and
        // `L_safe(q2) = a^ω`
        let contained = ak22_figure_2().safe_containment_relation();
        assert_eq!(
            contained.into_iter().collect::<Vec<_>>(),
            vec![(0, 0), (1, 1), (2, 0), (2, 2)]
        );
    }

    #[test]
    fn safe_containment_with_empty_safe_languages() {
        // 0 and 1 have empty safe languages, while that of 2 is `a^ω`
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
        // after reading `a` safely, 0 can read `b` and `c` safely forever, while 2 can only
        // read `c`
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
    fn simple_frontier_example() {
        // AK22, Example 3.10: `q2 ≤ q0`, so `H({q2}, {q0, q1})` and the frontier is `{{q0, q1}}`
        let dcw = ak22_figure_2();
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
    fn frontier_picks_smallest_of_strongly_equivalent_components() {
        // AK22, Figure 1: `H` relates `{q0}` and `{q1}` in both directions, so only the
        // component with the smaller state is kept
        let dcw = ak22_figure_1();
        assert!(!dcw.is_safe_centralized());
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
        // L(q1) is "finitely many `a`s" and L(q0) = a^ω + a^* b L(q1), so `H` only relates each
        // safe component to itself
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
            ])
            .into_dcw(0);
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
        let centralized = dcw.clone().into_safe_centralized();
        assert_eq!(sorted_transitions(&centralized), sorted_transitions(&dcw));
    }

    #[test]
    fn ak22_safe_centralization_example() {
        // AK22, Example 3.10: q2 is removed, and the `α`-transitions of q0 and q1 are
        // redirected to both q0 and q1. The result is nondeterministic on `c`.
        let dcw = ak22_figure_2();
        assert!(!dcw.is_safe_centralized());
        let ncw = dcw.into_safe_centralized();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_alpha_homogeneous());
        assert!(ncw.try_into_dcw().is_err());
    }

    #[test]
    fn safe_centralize_moves_initial_state_out_of_removed_component() {
        // as `q2 ≤ q0` but not `q2 ≤ q1`, the new initial state is q0
        let mut ncw = NCW::builder().with_edges(AK22_FIGURE_2).into_ncw(2);
        ncw.safe_centralize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
    }

    #[test]
    fn ak22_single_state_example() {
        // AK22, Figure 1, but starting in the removed q1, so the initial state is moved to q0
        let dcw = DCW::builder()
            .with_edges(AK22_FIGURE_1)
            .into_dcw(1)
            .into_safe_centralized()
            .try_into_dcw()
            .expect("the result is deterministic");
        assert_eq!(dcw.initial(), 0);
        assert_eq!(
            sorted_transitions(&dcw),
            vec![(0, 'a', false, 0), (0, 'b', true, 0)]
        );
        assert!(dcw.is_safe_centralized());
    }

    #[test]
    #[should_panic(expected = "safe deterministic")]
    fn safe_centralize_panics_if_not_safe_deterministic() {
        not_safe_deterministic_ncw().safe_centralize();
    }

    #[test]
    fn alpha_homogeneous() {
        assert!(!safe_and_alpha_on_same_letter_ncw().is_alpha_homogeneous());
        // several `α`-transitions on the same letter are fine
        let ncw = NCW::builder().with_edges(AK22_FIGURE_4).into_ncw(0);
        assert!(ncw.is_alpha_homogeneous());
    }
}
