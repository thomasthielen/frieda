use crate::automaton::Semantics;
use crate::core::{
    Color,
    alphabet::{Alphabet, Expression},
};
use crate::ts::{Deterministic, Shrinkable, Sproutable, StateColor};
use crate::{
    Automaton, DTS, NTS, Pointed, TransitionSystem, automaton::InfiniteWordAutomaton, ts::run,
};
use automata_core::Void;
use automata_core::alphabet::CharAlphabet;
use std::collections::{BTreeMap, BTreeSet};

// The steps of the minimization pipeline of [AK22] are implemented as further inherent methods
// on the co-Büchi automata below, one module per step.
//
// [AK22]: Bader Abu Radi, Orna Kupferman: Minimization and Canonization of GFG Transition-Based
//         Automata. Log. Methods Comput. Sci. 18(3) (2022)
mod centralize;
mod determinize;
mod minimize;
mod normalize;

use determinize::LanguageClasses;

#[cfg(test)]
mod ak22_examples;

/// Defines the [`Semantics`] of a deterministic co-Büchi automaton (DCW),
/// which is an acceptor of infinite words. It is the dual of [`super::BuchiCondition`]:
/// it considers the set of transitions which is taken infinitely often and accepts
/// iff *none* of those transitions are colored with `true`, i.e. iff transitions
/// colored `true` are taken only finitely often.
///
/// This type will rarely be used on its own, for the automaton that makes
/// use of it, see [`DCW`].
#[derive(Debug, Default, Clone, Eq, PartialEq, Hash, Copy)]
pub struct CoBuchiCondition;

impl<T: Deterministic<EdgeColor = bool>> Semantics<T, true> for CoBuchiCondition {
    type Observer = run::GreatestEdgeColor<T>;
    type Output = bool;
    fn evaluate(&self, observed: <Self::Observer as run::Observer<T>>::Current) -> Self::Output {
        !observed
    }
}

/// A deterministic co-Büchi automaton (DCW) is a deterministic automaton with
/// (transition-based) co-Büchi acceptance condition. It accepts a word if its run
/// takes `α`-transitions (i.e. ones that are labeled with `true`) only finitely often.
/// This is the dual of [`super::DBA`].
///
/// ### Naming of transitions
/// Following \[AK22\], the acceptance condition is a set `α` of transitions, which here
/// are the ones colored `true`. We call these the `α`-transitions, and the ones colored
/// `false` the `ᾱ`-transitions or *safe* transitions. The terms accepting/rejecting
/// are reserved for runs, not transitions: a run is accepting iff it takes
/// `α`-transitions only finitely often, and it is *safe* iff it takes none at all.
///
/// ### References
/// \[AK22\] Bader Abu Radi, Orna Kupferman: *Minimization and Canonization of GFG
/// Transition-Based Automata*. Log. Methods Comput. Sci. 18(3) (2022)
///
/// \[KS15\] Denis Kuperberg, Michal Skrzypczak: *On Determinisation of Good-for-Games
/// Automata*. ICALP (2) 2015: 299-310
pub type DCW<A = CharAlphabet, Q = Void, D = DTS<A, Q, bool>> =
    InfiniteWordAutomaton<A, CoBuchiCondition, Q, bool, true, D>;
/// Helper trait for creating a [`DCW`] from a given transition system.
pub type IntoDCW<T> = DCW<<T as TransitionSystem>::Alphabet, StateColor<T>, T>;

/// A (possibly) nondeterministic transition-based co-Büchi automaton (NCW). Unlike [`DCW`],
/// the backing transition system defaults to [`NTS`], which permits multiple `σ`-successors
/// from a state.
///
/// Since [`CoBuchiCondition`]'s [`Semantics`] impl requires `T: Deterministic`, a genuinely
/// nondeterministic `NCW` does not get `accepts`/`transform`.
pub type NCW<A = CharAlphabet, Q = Void, D = NTS<A, Q, bool>> =
    InfiniteWordAutomaton<A, CoBuchiCondition, Q, bool, true, D>;
/// Helper trait for creating an [`NCW`] from a given transition system.
pub type IntoNCW<T> = NCW<<T as TransitionSystem>::Alphabet, StateColor<T>, T>;

impl<A: Alphabet, Q: Color> DCW<A, Q> {
    /// Turns `self` into an [`NCW`] with the same states, transitions and initial state. The
    /// backing transition system is only reinterpreted as nondeterministic, so no state indices
    /// change.
    ///
    /// Following \[AK22\], every tDCW is a tNCW without nondeterministic choices, so this loses
    /// nothing. It is needed before applying transformations whose result may be
    /// nondeterministic, such as [`NCW::safe_centralize`], which are therefore only available on
    /// an [`NCW`]. See also [`Self::into_safe_centralized`], [`Self::into_safe_minimized`] and
    /// [`Self::into_minimized`].
    pub fn into_ncw(self) -> NCW<A, Q> {
        let (ts, initial, acceptance) = self.into_parts();
        NCW::from_parts_with_acceptance(ts.into_nondeterministic(), initial, acceptance)
    }

    /// Safe-centralizes `self` as a tNCW, see [`Self::into_ncw`] and [`NCW::safe_centralize`].
    /// The result may be nondeterministic, even though `self` is deterministic.
    pub fn into_safe_centralized(self) -> NCW<A, Q> {
        let mut ncw = self.into_ncw();
        ncw.safe_centralize();
        ncw
    }

    /// Safe-minimizes `self` as a tNCW, see [`Self::into_ncw`] and [`NCW::safe_minimize`]. The
    /// result may be nondeterministic, even though `self` is deterministic.
    pub fn into_safe_minimized(self) -> NCW<A, Q> {
        let mut ncw = self.into_ncw();
        ncw.safe_minimize();
        ncw
    }

    /// Minimizes `self` as a tNCW, see [`Self::into_ncw`] and [`NCW::minimize`]. The result is a
    /// minimal GFG-tNCW, which may be nondeterministic, even though `self` is deterministic.
    pub fn into_minimized(self) -> NCW<A, Q> {
        let mut ncw = self.into_ncw();
        ncw.minimize();
        ncw
    }
}

impl<A: Alphabet, Q: Color> NCW<A, Q> {
    /// Turns `self` into a [`DCW`] with the same states, transitions and initial state, if
    /// `self` is deterministic. Otherwise, `self` is returned unchanged as the error. This is the
    /// inverse of [`DCW::into_ncw`], and no state indices change.
    pub fn try_into_dcw(self) -> Result<DCW<A, Q>, Self> {
        let (ts, initial, acceptance) = self.into_parts();
        match ts.try_into_deterministic() {
            Ok(ts) => Ok(DCW::from_parts_with_acceptance(ts, initial, acceptance)),
            Err(ts) => Err(NCW::from_parts_with_acceptance(ts, initial, acceptance)),
        }
    }

    /// Minimizes `self`, which is assumed to be a GFG-tNCW whose states are all GFG, by applying
    /// the steps of \[AK22\] in order:
    /// 1. [`Self::semantically_determinize`], [`Self::remove_unreachable_states`] and
    ///    [`Self::remove_empty_states`], followed by [`Self::normalize`], which make `self` nice
    ///    \[AK22, Theorem 2.2\],
    /// 2. [`Self::safe_centralize`], which makes it safe-centralized and `α`-homogenous
    ///    \[AK22, Theorem 3.15\],
    /// 3. [`Self::safe_minimize`], which makes it safe-minimal \[AK22, Theorem 3.20\].
    ///
    /// Afterwards, `self` is a nice, safe-centralized and safe-minimal GFG-tNCW equivalent to the
    /// original one, and hence a minimal GFG-tNCW \[AK22, Theorem 3.6\]. The indices of the
    /// remaining states are unchanged.
    ///
    /// Missing transitions are treated as leading to an implicit rejecting sink, which is not
    /// counted as a state \[AK22, Remark 3.7\]. Hence, the result has no state with an empty
    /// language, unless `L(A) = ∅`, in which case it consists of the initial state only, without
    /// any transitions. Explicit rejecting sinks of the input are removed, so the size of the
    /// result only depends on the language.
    ///
    /// Unlike calling the steps one by one, the containment game (see
    /// [`Self::gfg_containment_relation`]) is solved only once, before step 1. All later steps
    /// keep the languages of the remaining states, so steps 2 and 3 reuse the resulting partition
    /// into language-equivalent states, and step 3 reuses the subsafe-equivalence relation
    /// computed in step 2 (see [`Self::subsafe_equivalence_relation`]).
    ///
    /// As steps 2 and 3 may introduce nondeterminism, this is only available on an [`NCW`]; use
    /// [`DCW::into_minimized`] to minimize a [`DCW`].
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]).
    ///
    /// # References
    /// \[AK22\] Bader Abu Radi, Orna Kupferman: *Minimization and Canonization of GFG
    /// Transition-Based Automata*. Log. Methods Comput. Sci. 18(3) (2022)
    pub fn minimize(&mut self) {
        self.minimize_reporting(|_| {});
    }

    /// Does the same as [`Self::minimize`], but calls `finished` with each
    /// [`MinimizationStep`] right after that step is done, e.g. for timing the individual steps
    /// including the reuse of results across them.
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]).
    pub fn minimize_reporting(&mut self, mut finished: impl FnMut(MinimizationStep)) {
        assert!(
            self.is_safe_deterministic(),
            "minimization requires a safe deterministic tNCW"
        );
        finished(MinimizationStep::SafeDeterminismCheck);
        let contained = self.gfg_containment_relation();
        finished(MinimizationStep::ContainmentGame);
        self.semantically_determinize_with(&contained);
        finished(MinimizationStep::SemanticDeterminization);
        self.remove_unreachable_states();
        finished(MinimizationStep::UnreachableRemoval);
        self.remove_empty_states();
        finished(MinimizationStep::EmptyStateRemoval);
        self.normalize();
        finished(MinimizationStep::Normalization);
        let classes = LanguageClasses::from_containment(&contained, self.state_indices());
        finished(MinimizationStep::LanguageClasses);
        let subsafe = self.safe_centralize_with(&classes);
        finished(MinimizationStep::SafeCentralization);
        self.safe_minimize_with(&subsafe);
        finished(MinimizationStep::SafeMinimization);
    }
}

/// The steps of [`NCW::minimize`], in the order they are executed, as reported by
/// [`NCW::minimize_reporting`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MinimizationStep {
    /// Checking that the input is safe deterministic.
    SafeDeterminismCheck,
    /// Solving the containment game once (see [`NCW::gfg_containment_relation`]), whose result
    /// is reused by all later steps.
    ContainmentGame,
    /// Removing non-covering transitions (see [`NCW::semantically_determinize`]).
    SemanticDeterminization,
    /// Removing unreachable states (see [`NCW::remove_unreachable_states`]).
    UnreachableRemoval,
    /// Removing states with an empty language (see [`NCW::remove_empty_states`]).
    EmptyStateRemoval,
    /// Recoloring safe transitions between safe components (see [`NCW::normalize`]).
    Normalization,
    /// Partitioning the states into language-equivalent classes from the containment relation.
    LanguageClasses,
    /// Making the automaton safe-centralized (see [`NCW::safe_centralize`]), reusing the
    /// language classes.
    SafeCentralization,
    /// Making the automaton safe-minimal (see [`NCW::safe_minimize`]), reusing the
    /// subsafe-equivalence relation computed during safe centralization.
    SafeMinimization,
}

/// This impl applies to both [`DCW`] and [`NCW`], as they are both instantiations of
/// [`Automaton`] with a [`CoBuchiCondition`] and `bool`-colored transitions, differing
/// only in the (default) type of the backing transition system.
impl<A, Q, D> Automaton<A, CoBuchiCondition, Q, bool, D, true, true>
where
    A: Alphabet,
    Q: Color,
    D: TransitionSystem<Alphabet = A, StateColor = Q, EdgeColor = bool>,
{
    /// Returns `true` iff `self` is safe deterministic. A (t)NCW is safe deterministic if
    /// removing its `α`-transitions (colored `true`) removes all nondeterministic choices.
    /// Formally, this means that for every state `q` and symbol `σ`, there is at most one
    /// `ᾱ`-transition (i.e. a safe transition, colored `false`) labeled `σ` leaving `q`;
    /// there may still be arbitrarily many `α`-transitions on `q` and `σ`.
    ///
    /// Note that every genuinely deterministic transition system (such as the one backing a
    /// [`DCW`]) is trivially safe deterministic, since it has at most one `σ`-transition from
    /// `q` at all.
    pub fn is_safe_deterministic(&self) -> bool {
        self.state_indices().all(|q| {
            let mut safe_successors_seen = BTreeSet::new();
            self.transitions_from(q)
                .filter(|&(_, _, color, _)| !color)
                .all(|(_, sym, _, _)| safe_successors_seen.insert(sym))
        })
    }

    /// Removes all states of `self` that are not reachable from the initial state, together with
    /// all transitions leaving or entering them, and returns the removed states with their
    /// colors. The indices of the remaining states are unchanged.
    ///
    /// Following \[AK22, Theorem 2.2\], this is the step after [`Self::semantically_determinize`]:
    /// removing unreachable states affects neither the language nor the GFGness of `self`, since
    /// no run from the initial state ever visits them. It also preserves the other properties
    /// obtained so far: the languages of the remaining states only depend on states reachable
    /// from them, which are all kept, so safe determinism, the GFGness of each state and
    /// semantic determinism all carry over. Doing this after semantic determinization matters,
    /// as removing non-covering transitions may make further states unreachable.
    pub fn remove_unreachable_states(&mut self) -> Vec<(D::StateIndex, Q)>
    where
        D: Shrinkable,
    {
        self.trim()
    }

    /// Removes all states `q` of `self` with `L(A^q) = ∅`, together with all transitions leaving
    /// or entering them, and returns the removed states with their colors. The indices of the
    /// remaining states are unchanged. The initial state is never removed: if `L(A) = ∅`, it is
    /// kept as the only state, without any transitions.
    ///
    /// Missing transitions are treated as leading to an implicit rejecting sink, which is not
    /// counted as a state \[AK22, Remark 3.7\]. Removing the states with an empty language makes
    /// `self` follow this convention, so that the size of a minimal GFG-tNCW does not depend on
    /// whether its input contained an explicit rejecting sink.
    ///
    /// A run is accepting iff it eventually stays on a cycle of safe transitions, so `L(A^q)` is
    /// nonempty iff `q` can reach such a cycle; this does not require `self` to be GFG. Removing
    /// the other states affects neither the language nor the GFGness of any remaining state, and
    /// it preserves the properties obtained by \[AK22, Theorem 2.2\]: only transitions disappear,
    /// which keeps safe determinism and normality, and if some `σ`-successor of a state has an
    /// empty language, then all of them have, so semantic determinism is kept as well. Also, no
    /// state becomes unreachable, since a state that is only reachable via states with an empty
    /// language has an empty language itself.
    pub fn remove_empty_states(&mut self) -> Vec<(D::StateIndex, Q)>
    where
        D: Shrinkable,
    {
        // states on a cycle of safe transitions, from which nonempty is propagated backwards
        let mut nonempty: BTreeSet<D::StateIndex> = self
            .edge_color_restricted(false, false)
            .sccs()
            .sccs_iter()
            .filter(|scc| scc.is_nontransient())
            .flat_map(|scc| scc.iter().copied().collect::<Vec<_>>())
            .collect();
        let mut predecessors: BTreeMap<D::StateIndex, Vec<D::StateIndex>> = BTreeMap::new();
        for q in self.state_indices() {
            for (_, _, _, target) in self.transitions_from(q) {
                predecessors.entry(target).or_default().push(q);
            }
        }
        let mut stack: Vec<D::StateIndex> = nonempty.iter().copied().collect();
        while let Some(p) = stack.pop() {
            for &q in predecessors.get(&p).into_iter().flatten() {
                if nonempty.insert(q) {
                    stack.push(q);
                }
            }
        }

        let initial = self.initial();
        if !nonempty.contains(&initial) {
            self.remove_edges_from(initial)
                .expect("the initial state exists");
        }
        let empty: Vec<D::StateIndex> = self
            .state_indices()
            .filter(|q| *q != initial && !nonempty.contains(q))
            .collect();
        empty
            .into_iter()
            .map(|q| {
                (
                    q,
                    self.remove_state(q)
                        .expect("state was obtained from state_indices"),
                )
            })
            .collect()
    }

    /// Adds the given `edges` to `self`, but adds each transition `⟨q, σ, c, p⟩` matched by them
    /// only once, as the backing transition system keeps edges that match the same transition
    /// apart. Edges whose expression only matches transitions that were not added before are
    /// added as they are; otherwise, an edge is replaced by one edge for each symbol of a
    /// transition that is new. Transitions that were already in `self` before are not taken
    /// into account.
    ///
    /// This is needed whenever several edges are mapped to the same source, color and target,
    /// whose expressions may overlap without being equal if they can match several symbols.
    pub(super) fn add_edges_without_duplicates(
        &mut self,
        edges: impl IntoIterator<Item = (D::StateIndex, A::Expression, bool, D::StateIndex)>,
    ) where
        D: Sproutable,
    {
        let mut added = BTreeSet::new();
        for (source, expression, color, target) in edges {
            let symbols: Vec<A::Symbol> = expression.symbols().collect();
            let new: Vec<A::Symbol> = symbols
                .iter()
                .copied()
                .filter(|&sym| added.insert((source, sym, color, target)))
                .collect();
            if new.len() == symbols.len() {
                self.add_edge((source, expression, color, target));
                continue;
            }
            for sym in new {
                let expression = self.alphabet().make_expression(sym);
                self.add_edge((source, expression, color, target));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ak22_examples::{
        AK22_FIGURE_2, AK22_FIGURE_4, ak22_figure_1, ak22_figure_2, ak22_figure_2_extended,
        language_equivalent, non_covering_ncw, not_safe_deterministic_ncw,
        safe_and_alpha_on_same_letter_ncw, sorted_transitions,
    };
    use super::{DCW, LanguageClasses, NCW};
    use crate::random::{generate_random_dcw, generate_random_ncw};
    use crate::ts::TSBuilder;
    use crate::{Pointed, TransitionSystem};
    use automata_core::{Void, upw};
    use rand::{SeedableRng, rngs::StdRng};

    #[test]
    fn dcws() {
        // complement of the `dbas` test in `automaton.rs`
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', true, 1),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', false, 0),
            ])
            .into_dcw(0);
        assert!(!dcw.accepts(upw!("abb")));
        assert!(dcw.accepts(upw!("b")));
        assert!(!dcw.accepts(upw!("a")));
        assert!(dcw.accepts(upw!("aab", "b")));

        assert!(!dcw.is_empty());
    }

    #[test]
    fn ncw_is_a_transition_system() {
        let nts = TSBuilder::<Void, bool, true>::without_state_colors()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_nts();
        let ncw = NCW::from_parts(nts, 0);
        assert_eq!(ncw.edges_from(0).unwrap().count(), 2);
    }

    #[test]
    fn dcw_into_ncw() {
        let dcw = DCW::builder().with_edges(AK22_FIGURE_2).into_dcw(2);
        let ncw = dcw.clone().into_ncw();
        assert_eq!(ncw.initial(), 2);
        assert_eq!(
            ncw.state_indices().collect::<Vec<_>>(),
            dcw.state_indices().collect::<Vec<_>>()
        );
        assert_eq!(sorted_transitions(&ncw), sorted_transitions(&dcw));
    }

    #[test]
    fn safe_deterministic() {
        assert!(ak22_figure_1().is_safe_deterministic());
        assert!(safe_and_alpha_on_same_letter_ncw().is_safe_deterministic());
        assert!(!not_safe_deterministic_ncw().is_safe_deterministic());
    }

    #[test]
    fn remove_unreachable_states_removes_incoming_transitions() {
        // 2 and 3 are unreachable, and so are their transitions into the reachable states.
        // State 1 is only reachable via an `α`-transition.
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
                (2, 'a', false, 0),
                (2, 'b', false, 3),
                (3, 'a', true, 2),
                (3, 'b', false, 1),
            ])
            .into_ncw(0);
        let mut removed: Vec<u32> = ncw
            .remove_unreachable_states()
            .into_iter()
            .map(|(q, _)| q)
            .collect();
        removed.sort();
        assert_eq!(removed, vec![2, 3]);
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', false, 0),
                (0, 'b', true, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ]
        );
    }

    #[test]
    fn remove_empty_states_keeps_states_reaching_safe_cycles() {
        // 1 and 2 never reach a safe cycle, while 3 reaches the safe cycle of 4 via an
        // `α`-transition
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (0, 'c', true, 3),
                (1, 'a', true, 2),
                (1, 'b', true, 1),
                (2, 'a', true, 2),
                (3, 'a', true, 4),
                (4, 'a', false, 4),
            ])
            .into_ncw(0);
        let mut removed: Vec<u32> = ncw
            .remove_empty_states()
            .into_iter()
            .map(|(q, _)| q)
            .collect();
        removed.sort();
        assert_eq!(removed, vec![1, 2]);
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', false, 0),
                (0, 'c', true, 3),
                (3, 'a', true, 4),
                (4, 'a', false, 4),
            ]
        );
    }

    #[test]
    fn remove_empty_states_keeps_initial_state_of_empty_language() {
        let mut ncw = NCW::builder()
            .with_edges([(0, 'a', true, 0), (0, 'b', false, 1), (1, 'a', true, 0)])
            .into_ncw(0);
        assert_eq!(ncw.remove_empty_states(), vec![(1, Void)]);
        assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![0]);
        assert!(sorted_transitions(&ncw).is_empty());
    }

    #[test]
    fn minimize_ignores_explicit_rejecting_sink() {
        // both accept `a^ω`, once with an explicit rejecting sink 1 and once with a missing
        // `b`-transition, AK22, Remark 3.7
        let with_sink = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', true, 1),
                (1, 'b', true, 1),
            ])
            .into_dcw(0);
        let without_sink = NCW::builder()
            .with_edges([(0, 'a', false, 0), (1, 'b', true, 1)])
            .into_ncw(0);
        for mut ncw in [with_sink.into_ncw(), without_sink] {
            ncw.minimize();
            assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![0]);
            assert_eq!(sorted_transitions(&ncw), vec![(0, 'a', false, 0)]);
        }
    }

    #[test]
    fn nice_after_full_pipeline() {
        // semantic determinization makes 2 unreachable, and the remaining safe transition
        // `⟨0, a, 1⟩` connects the safe components {0} and {1}
        let mut ncw = non_covering_ncw();
        assert!(ncw.remove_unreachable_states().is_empty());
        ncw.semantically_determinize();
        assert_eq!(ncw.remove_unreachable_states(), vec![(2, Void)]);
        assert!(!ncw.is_normal());
        ncw.normalize();
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.remove_unreachable_states().is_empty());
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', true, 1),
                (0, 'b', false, 0),
                (1, 'a', false, 1),
                (1, 'b', false, 1),
            ]
        );
    }

    #[test]
    fn minimize_ak22_figure_2_extended_step_by_step() {
        let mut ncw = ak22_figure_2_extended();
        assert!(ncw.is_safe_deterministic());
        assert!(!ncw.is_semantically_deterministic());
        assert!(!ncw.is_alpha_homogeneous());

        ncw.semantically_determinize();
        assert!(ncw.is_semantically_deterministic());
        let mut removed: Vec<u32> = ncw
            .remove_unreachable_states()
            .into_iter()
            .map(|(q, _)| q)
            .collect();
        removed.sort();
        assert_eq!(removed, vec![3, 4]);
        assert!(!ncw.is_normal());
        ncw.normalize();
        assert!(ncw.is_normal());
        assert!(!ncw.is_safe_centralized());

        ncw.safe_centralize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
        // AK22, Proposition 3.14
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.remove_unreachable_states().is_empty());
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_alpha_homogeneous());

        // AK22, Example 3.21: Figure 4 is already safe-minimal
        let mut minimized = ak22_figure_2_extended();
        minimized.minimize();
        assert_eq!(minimized.initial(), 0);
        assert_eq!(sorted_transitions(&minimized), AK22_FIGURE_4.to_vec());
    }

    #[test]
    fn minimize_needs_every_step() {
        // all states accept "finitely many `b`s" with safe language `a^ω`. Normalization
        // recolors `⟨2, a, 0⟩`, safe centralization removes {2}, and safe minimization merges
        // 0 and 1, resulting in the single-state `A_fm` of AK22, Figure 1.
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'b', true, 0),
                (1, 'a', false, 0),
                (1, 'b', true, 1),
                (2, 'a', false, 0),
                (2, 'b', true, 2),
            ])
            .into_dcw(2);
        let minimized = dcw
            .into_minimized()
            .try_into_dcw()
            .expect("the result is deterministic");
        assert_eq!(minimized.initial(), 0);
        assert_eq!(
            sorted_transitions(&minimized),
            vec![(0, 'a', false, 0), (0, 'b', true, 0)]
        );
    }

    #[test]
    fn minimize_ak22_figure_2_as_dcw_into_nondeterministic_ncw() {
        let ncw = ak22_figure_2().into_minimized();
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
        assert!(ncw.try_into_dcw().is_err());
    }

    #[test]
    fn minimize_reuses_relations_correctly() {
        // the relations reused by `minimize` agree with freshly computed ones after every step,
        // and the result agrees with calling the steps one by one
        for seed in 0..30 {
            let input = generate_random_ncw(2, 10, 0.3, 0.5, &mut StdRng::seed_from_u64(seed));

            let mut stepwise = input.clone();
            stepwise.semantically_determinize();
            stepwise.remove_unreachable_states();
            stepwise.remove_empty_states();
            stepwise.normalize();
            stepwise.safe_centralize();
            stepwise.safe_minimize();

            let mut ncw = input.clone();
            let contained = ncw.gfg_containment_relation();
            ncw.semantically_determinize_with(&contained);
            ncw.remove_unreachable_states();
            ncw.remove_empty_states();
            ncw.normalize();
            let classes = LanguageClasses::from_containment(&contained, ncw.state_indices());
            assert_eq!(
                classes,
                LanguageClasses::from_containment(
                    &ncw.gfg_containment_relation(),
                    ncw.state_indices()
                ),
                "language classes changed during the first step for seed {seed}"
            );
            let subsafe = ncw.safe_centralize_with(&classes);
            assert_eq!(
                subsafe,
                ncw.subsafe_equivalence_relation(),
                "subsafe-equivalence changed during safe centralization for seed {seed}"
            );
            ncw.safe_minimize_with(&subsafe);
            assert!(ncw.is_safe_minimal());
            assert_eq!(ncw.initial(), stepwise.initial());
            assert_eq!(sorted_transitions(&ncw), sorted_transitions(&stepwise));

            let mut minimized = input;
            minimized.minimize();
            assert_eq!(sorted_transitions(&minimized), sorted_transitions(&ncw));
        }
    }

    #[test]
    fn minimize_preserves_language() {
        // random GFG-tNCWs and tDCWs, the latter over three letters and with so many
        // `α`-transitions that 12 of them contain states with an empty language. The result
        // has the language of the input, has the properties of AK22, Theorem 3.20, and cannot
        // be minimized further.
        for seed in 0..30 {
            let rng = &mut StdRng::seed_from_u64(seed);
            for input in [
                generate_random_ncw(2, 10, 0.3, 0.5, rng),
                generate_random_dcw(3, 8, 0.8, rng).into_ncw(),
            ] {
                let mut minimized = input.clone();
                minimized.minimize();
                assert!(
                    language_equivalent(&input, &minimized),
                    "language changed for seed {seed}"
                );
                assert!(minimized.is_safe_deterministic());
                assert!(minimized.is_semantically_deterministic());
                assert!(minimized.is_normal());
                assert!(minimized.is_safe_centralized());
                assert!(minimized.is_safe_minimal());
                assert!(minimized.is_alpha_homogeneous());

                let mut again = minimized.clone();
                again.minimize();
                assert_eq!(
                    sorted_transitions(&again),
                    sorted_transitions(&minimized),
                    "minimizing again changed the result for seed {seed}"
                );
            }
        }
    }

    #[test]
    fn language_equivalent_detects_different_languages() {
        // the helper used by `minimize_preserves_language` must not accept everything: `a^ω`
        // and "finitely many `b`s" differ, and so do `a^ω` and the empty language
        let a_omega = NCW::builder()
            .with_edges([(0, 'a', false, 0), (0, 'b', true, 1), (1, 'b', true, 1)])
            .into_ncw(0);
        let empty = NCW::builder()
            .with_edges([(0, 'a', true, 0), (0, 'b', true, 0)])
            .into_ncw(0);
        let finitely_many_b = ak22_figure_1().into_ncw();
        assert!(language_equivalent(&a_omega, &a_omega));
        assert!(language_equivalent(&finitely_many_b, &finitely_many_b));
        assert!(!language_equivalent(&a_omega, &finitely_many_b));
        assert!(!language_equivalent(&finitely_many_b, &a_omega));
        assert!(!language_equivalent(&a_omega, &empty));
    }

    #[test]
    #[should_panic(expected = "safe deterministic")]
    fn minimize_panics_if_not_safe_deterministic() {
        not_safe_deterministic_ncw().minimize();
    }
}
