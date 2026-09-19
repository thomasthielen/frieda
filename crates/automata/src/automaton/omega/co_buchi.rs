use crate::automaton::Semantics;
use crate::ts::{Deterministic, StateColor};
use crate::{DTS, NTS, TransitionSystem, automaton::InfiniteWordAutomaton, ts::run};
use automata_core::Void;
use automata_core::alphabet::CharAlphabet;

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
/// (transition-based) co-Büchi acceptance condition. It accepts a word if it has a
/// successful infinite run that takes an accepting transition (i.e. one that is
/// labeled with `true`) only finitely often. This is the dual of [`super::DBA`].
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

#[cfg(test)]
mod tests {
    use super::{DCW, NCW};
    use crate::ts::TSBuilder;
    use automata_core::{Void, upw};

    #[test]
    fn dcws() {
        // dual of the `dbas` test in `automaton.rs`: accepts iff eventually only `a`s
        // are read, i.e. co-Büchi acceptance of "(a+b)*a^omega" via a `true`-colored
        // (rejecting) self-loop on `b`.
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 0),
                (1, 'a', false, 1),
                (1, 'b', true, 1),
            ])
            .into_dcw(0);
        assert!(dcw.accepts(upw!("a")));
        assert!(!dcw.accepts(upw!("b")));
        assert!(dcw.accepts(upw!("ab", "a")));
        assert!(!dcw.accepts(upw!("a", "b")));
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
}
