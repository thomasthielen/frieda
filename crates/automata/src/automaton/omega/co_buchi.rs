use crate::automaton::Semantics;
use crate::core::{Color, alphabet::Alphabet};
use crate::ts::{Deterministic, Shrinkable, Sproutable, StateColor};
use crate::{Automaton, DTS, NTS, TransitionSystem, automaton::InfiniteWordAutomaton, ts::run};
use automata_core::Void;
use automata_core::alphabet::CharAlphabet;
use std::collections::BTreeSet;

// The steps of the minimization pipeline of [AK22] are implemented as further inherent methods
// on the co-Büchi automata below, one module per step.
mod centralize;
mod determinize;
mod minimize;
mod normalize;

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
    /// This is needed before applying transformations that may introduce nondeterminism, such as
    /// [`NCW::safe_centralize`], which the deterministic transition system backing a [`DCW`]
    /// cannot represent.
    pub fn into_ncw(self) -> NCW<A, Q> {
        let (ts, initial, acceptance) = self.into_parts();
        NCW::from_parts_with_acceptance(ts.into_nondeterministic(), initial, acceptance)
    }
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

    /// Minimizes `self`, which is assumed to be a GFG-tNCW whose states are all GFG, by applying
    /// the steps of \[AK22\] in order:
    /// 1. [`Self::semantically_determinize`] and [`Self::remove_unreachable_states`], followed by
    ///    [`Self::normalize`], which make `self` nice \[AK22, Theorem 2.2\],
    /// 2. [`Self::safe_centralize`], which makes it safe-centralized and `α`-homogenous
    ///    \[AK22, Theorem 3.15\],
    /// 3. [`Self::safe_minimize`], which makes it safe-minimal \[AK22, Theorem 3.20\].
    ///
    /// Afterwards, `self` is a nice, safe-centralized and safe-minimal GFG-tNCW equivalent to the
    /// original one, and hence a minimal GFG-tNCW \[AK22, Theorem 3.6\]. The indices of the
    /// remaining states are unchanged.
    ///
    /// # Panics
    /// If `self` is not safe deterministic (see [`Self::is_safe_deterministic`]), or if one of
    /// the steps introduces nondeterminism but the transition system backing `self` is
    /// deterministic, as is the case for a [`DCW`]. This may happen even if the minimal result is
    /// deterministic, since [`Self::safe_centralize`] redirects `α`-transitions to all equivalent
    /// states it keeps. Use [`DCW::into_ncw`] first to minimize a [`DCW`].
    pub fn minimize(&mut self)
    where
        D: Shrinkable + Sproutable,
    {
        assert!(
            self.is_safe_deterministic(),
            "minimization requires a safe deterministic tNCW"
        );
        self.semantically_determinize();
        self.remove_unreachable_states();
        self.normalize();
        self.safe_centralize();
        self.safe_minimize();
    }
}

#[cfg(test)]
mod tests {
    use super::ak22_examples::{AK22_FIGURE_2, AK22_FIGURE_4, sorted_transitions};
    use super::{DCW, NCW};
    use crate::ts::TSBuilder;
    use crate::{Pointed, TransitionSystem};
    use automata_core::{Void, upw};

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
        use crate::TransitionSystem;
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
    fn dcw_is_safe_deterministic() {
        // every genuinely deterministic transition system is trivially safe deterministic.
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(0);
        assert!(dcw.is_safe_deterministic());
    }

    #[test]
    fn ncw_safe_deterministic() {
        let ncw = NCW::builder()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_ncw(0);
        assert!(ncw.is_safe_deterministic());
    }

    #[test]
    fn ncw_not_safe_deterministic() {
        let ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', false, 2),
                (1, 'a', false, 1),
                (2, 'a', false, 2),
            ])
            .into_ncw(0);
        assert!(!ncw.is_safe_deterministic());
    }

    #[test]
    fn remove_unreachable_states_after_semantic_determinization() {
        // same automaton as in `semantically_determinize_removes_non_covering_transition`:
        // removing the non-covering transition `⟨0, a, 2⟩` makes state 2 unreachable
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', true, 2),
                (0, 'b', false, 0),
                (1, 'a', false, 1),
                (1, 'b', false, 1),
                (2, 'a', true, 2),
                (2, 'b', true, 2),
            ])
            .into_ncw(0);
        assert!(ncw.remove_unreachable_states().is_empty());
        ncw.semantically_determinize();
        assert_eq!(ncw.remove_unreachable_states(), vec![(2, Void)]);
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', false, 1),
                (0, 'b', false, 0),
                (1, 'a', false, 1),
                (1, 'b', false, 1),
            ]
        );
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
    }

    #[test]
    fn remove_unreachable_states_removes_incoming_transitions() {
        // states 2 and 3 are unreachable from 0, and the transitions `⟨2, a, 0⟩` and
        // `⟨3, b, 1⟩` into reachable states must be removed along with them. State 1 is only
        // reachable via an `α`-transition, and must be kept.
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
    fn nice_after_full_pipeline() {
        // same automaton as in `semantically_determinize_removes_non_covering_transition`:
        // after semantic determinization and removal of unreachable states, the safe transition
        // `⟨0, a, 1⟩` still connects the safe components {0} and {1}
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', true, 2),
                (0, 'b', false, 0),
                (1, 'a', false, 1),
                (1, 'b', false, 1),
                (2, 'a', true, 2),
                (2, 'b', true, 2),
            ])
            .into_ncw(0);
        ncw.semantically_determinize();
        ncw.remove_unreachable_states();
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
    fn safe_centralized_after_full_pipeline() {
        // AK22, Figure 2 with two additional states, such that every step of the pipeline has an
        // effect:
        // - the `α`-transition `⟨q0, a, q3⟩` into the empty state 3 is not covering, and removing
        //   it makes state 3 unreachable, just like the state 4, which is unreachable anyway,
        // - the new initial state 5 reads every letter safely into q0, so `L(q5) = L(q0)` (all
        //   states of Figure 2 are equivalent), but these safe transitions connect the safe
        //   components {q5} and {q0, q1}, so they are recolored during normalization,
        // - safe centralization then removes q5 and q2, and moves the initial state to q0.
        let mut ncw = NCW::builder()
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
            .into_ncw(5);
        assert!(ncw.is_safe_deterministic());
        assert!(!ncw.is_semantically_deterministic());
        // q0 has a safe and an `α`-transition on `a`
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
        // the result is nice, safe-centralized and `α`-homogenous [AK22, Proposition 3.14]
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
        assert!(ncw.remove_unreachable_states().is_empty());
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_centralized());
        assert!(ncw.is_alpha_homogeneous());

        // AK22, Example 3.21: q0 and q1 differ in their safe languages, so safe minimization
        // changes nothing
        assert!(ncw.is_safe_minimal());
        ncw.safe_minimize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
    }

    #[test]
    fn minimize_ak22_figure_2_with_additional_states() {
        // same automaton as in `safe_centralized_after_full_pipeline`, minimized in one go
        let mut ncw = NCW::builder()
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
            .into_ncw(5);
        ncw.minimize();
        assert_eq!(ncw.initial(), 0);
        assert_eq!(ncw.state_indices().collect::<Vec<_>>(), vec![0, 1]);
        assert_eq!(sorted_transitions(&ncw), AK22_FIGURE_4.to_vec());
        assert!(ncw.is_safe_minimal());
    }

    #[test]
    fn minimize_needs_every_step() {
        // all states accept "finitely many `b`s" and have the safe language `a^ω`, so they are
        // strongly-equivalent:
        // - the safe transition `⟨2, a, 0⟩` connects the safe components {2} and {0, 1}, so it is
        //   recolored during normalization,
        // - safe centralization then keeps only {0, 1}, which contains the smallest state,
        // - safe minimization merges 0 and 1.
        // Safe centralization redirects the `α`-transitions on `b` to both 0 and 1, so the `DCW`
        // must be turned into an `NCW` first, even though the result is deterministic again.
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
        let mut ncw = dcw.clone().into_ncw();
        ncw.minimize();
        assert_eq!(ncw.initial(), 0);
        // the minimal tDCW for "finitely many `b`s", i.e. `A_fm` of AK22, Figure 1 after safe
        // centralization
        assert_eq!(
            sorted_transitions(&ncw),
            vec![(0, 'a', false, 0), (0, 'b', true, 0)]
        );
        let words = [upw!("a"), upw!("b"), upw!("ba", "a"), upw!("a", "ab")];
        let accepted: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();
        assert_eq!(accepted, vec![true, false, true, false]);
    }

    #[test]
    #[should_panic(expected = "deterministic transition system")]
    fn minimize_panics_on_dcw_if_intermediate_step_is_nondeterministic() {
        // same automaton as in `minimize_needs_every_step`, without turning it into an `NCW`
        let mut dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'b', true, 0),
                (1, 'a', false, 0),
                (1, 'b', true, 1),
                (2, 'a', false, 0),
                (2, 'b', true, 2),
            ])
            .into_dcw(2);
        dcw.minimize();
    }

    #[test]
    #[should_panic(expected = "safe deterministic")]
    fn minimize_panics_if_not_safe_deterministic() {
        // same automaton as in `ncw_not_safe_deterministic_with_two_safe_transitions`
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', false, 2),
                (1, 'a', false, 1),
                (2, 'a', false, 2),
            ])
            .into_ncw(0);
        ncw.minimize();
    }
}
