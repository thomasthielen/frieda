use crate::automaton::Semantics;
use crate::core::{Color, alphabet::Alphabet};
use crate::ts::{Deterministic, StateColor};
use crate::{Automaton, DTS, NTS, TransitionSystem, automaton::InfiniteWordAutomaton, ts::run};
use automata_core::Void;
use automata_core::alphabet::CharAlphabet;
use std::collections::BTreeSet;

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
/// takes `α`-transitions only finitely often. This is the dual of [`super::DBA`].
/// 
/// ### Naming of transitions
/// Following \[RK22\], the acceptance condition is a set `α` of transitions, which here
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

/// This impl applies to both [`DCW`] and [`NCW`], as they are both instantiations of
/// [`Automaton`] with a [`CoBuchiCondition`] and `bool`-colored transitions, differing
/// only in the (default) type of the backing transition system.
impl<A, Q, D> Automaton<A, CoBuchiCondition, Q, bool, D, true, true>
where
    A: Alphabet,
    Q: Color,
    D: TransitionSystem<Alphabet = A, StateColor = Q, EdgeColor = bool>,
{
    /// Returns `true` iff `self` is safe deterministic. Following \[RK22\], a (t)NCW is safe
    /// deterministic if removing its `α`-transitions (colored `true`, see [`DCW`] for the
    /// naming) removes all nondeterministic choices. Formally, this means that for every
    /// state `q` and symbol `σ`, there is at most one `ᾱ`-transition (i.e. a safe
    /// transition, colored `false`) labeled `σ` leaving `q`; there may still be arbitrarily
    /// many `α`-transitions on `q` and `σ`.
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
}

#[cfg(test)]
mod tests {
    use super::{DCW, NCW};
    use crate::ts::TSBuilder;
    use automata_core::{Void, upw};

    #[test]
    fn dcws() {
        // same automaton as in the `dbas` test in `automaton.rs`, but with co-Büchi
        // semantics: the `true`-colored transitions are the ones reading `a`, so this
        // accepts iff only finitely many `a`s are read, i.e. exactly the complement of
        // the language of the DBA.
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
    }

    #[test]
    fn ncw_is_a_transition_system() {
        use crate::TransitionSystem;

        // `NCW` must support genuine nondeterminism (two `a`-labeled edges from state 0)
        // and expose `TransitionSystem` directly, without needing `.accepts()`.
        let nts = TSBuilder::<Void, bool, true>::without_state_colors()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_nts();
        let ncw = NCW::from_parts(nts, 0);
        assert_eq!(ncw.edges_from(0).unwrap().count(), 2);
    }

    #[test]
    fn ncw_builder_into_ncw() {
        use crate::TransitionSystem;

        // same automaton as in `ncw_is_a_transition_system`, built via `into_ncw`
        let ncw = NCW::builder()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_ncw(0);
        assert_eq!(ncw.edges_from(0).unwrap().count(), 2);
    }

    #[test]
    fn into_ncw_does_not_check_safe_determinism() {
        // `into_ncw` is a plain constructor: two safe `a`-transitions leave state 0, and it
        // must still build the automaton instead of rejecting it.
        let ncw = NCW::builder()
            .with_edges([(0, 'a', false, 1), (0, 'a', false, 2), (1, 'a', false, 1), (2, 'a', false, 2)])
            .into_ncw(0);
        assert!(!ncw.is_safe_deterministic());
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
    fn ncw_safe_deterministic_despite_nondeterminism_on_alpha_transitions() {
        // two `a`-labeled edges leave state 0, but only one of them (to state 1) is
        // `false`-colored/safe; the other is a `true`-colored `α`-transition. Removing
        // the `α`-transition leaves a deterministic automaton, so this is safe deterministic.
        let nts = TSBuilder::<Void, bool, true>::without_state_colors()
            .with_edges([(0, 'a', true, 0), (0, 'a', false, 1), (1, 'a', false, 1)])
            .into_nts();
        let ncw = NCW::from_parts(nts, 0);
        assert!(ncw.is_safe_deterministic());
    }

    #[test]
    fn ncw_not_safe_deterministic_with_two_safe_transitions() {
        // state 0 has two `false`-colored (safe) `a`-transitions, to states 1 and 2.
        // Removing `α`-transitions does not resolve this choice, so it is not safe
        // deterministic, even though the automaton has no `α`-transitions at all.
        let nts = TSBuilder::<Void, bool, true>::without_state_colors()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', false, 2),
                (1, 'a', false, 1),
                (2, 'a', false, 2),
            ])
            .into_nts();
        let ncw = NCW::from_parts(nts, 0);
        assert!(!ncw.is_safe_deterministic());
    }
}
