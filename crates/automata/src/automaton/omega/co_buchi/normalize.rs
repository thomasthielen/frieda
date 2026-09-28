//! Safe components and normalization \[RK22, Theorem 2.2\], \[KS15, Lemma 46\].
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
    /// Returns, for every state of `self`, the index of its safe component. Following \[RK22\],
    /// the safe components of a tNCW are the SCCs of the graph obtained by removing all
    /// `α`-transitions (colored `true`, see [`DCW`](super::DCW) for the naming). Two states are
    /// in the same safe component iff they are mapped to the same index.
    pub(super) fn safe_component_indices(&self) -> BTreeMap<D::StateIndex, usize> {
        self.edge_color_restricted(false, false)
            .sccs()
            .sccs_iter()
            .enumerate()
            .flat_map(|(i, scc)| scc.iter().map(move |&q| (q, i)))
            .collect()
    }

    /// Returns `true` iff `self` is normal. Following \[RK22\], a tNCW is normal if there are no
    /// `ᾱ`-transitions (i.e. safe transitions, colored `false`) connecting different safe
    /// components. That is, whenever there is a path of `ᾱ`-transitions from `q` to `s`, there
    /// is also a path of `ᾱ`-transitions from `s` to `q`.
    pub fn is_normal(&self) -> bool {
        let component = self.safe_component_indices();
        self.state_indices().all(|q| {
            self.transitions_from(q)
                .filter(|&(_, _, color, _)| !color)
                .all(|(_, _, _, target)| component[&q] == component[&target])
        })
    }

    /// Normalizes `self` by turning every `ᾱ`-transition (i.e. safe transition, colored `false`)
    /// that connects different safe components into an `α`-transition (colored `true`),
    /// following \[KS15, Lemma 46\]. The safe components are computed once, before any
    /// transition is recolored; recoloring then leaves the safe components unchanged, as it
    /// only removes edges of the safe graph that lie on no cycle of it. Afterwards,
    /// [`Self::is_normal`] holds.
    ///
    /// For every word, the accepting runs of `self` are the same before and after: a run that
    /// takes a recolored transition infinitely often leaves a safe component infinitely often,
    /// and since the safe components form a DAG, it must then also take infinitely many
    /// `α`-transitions that were already there. Hence, the language and the GFGness of every
    /// state are unchanged \[RK22, Theorem 2.2\]. As no transition or state is removed, and
    /// transitions only ever change from safe to `α`, reachability, safe determinism and
    /// semantic determinism are preserved as well. This is the last step towards a *nice*
    /// GFG-tNCW, after [`Self::remove_unreachable_states`].
    ///
    /// If a recolored transition coincides with an `α`-transition that is already present, only
    /// one copy of it is kept.
    pub fn normalize(&mut self)
    where
        D: Shrinkable + Sproutable,
    {
        let component = self.safe_component_indices();
        let states: Vec<D::StateIndex> = self.state_indices().collect();
        for q in states {
            let edges = self
                .remove_edges_from(q)
                .expect("state was obtained from state_indices");
            let normalized: BTreeSet<_> = edges
                .into_iter()
                .map(|(source, expression, color, target)| {
                    let color = color || component[&source] != component[&target];
                    (source, expression, color, target)
                })
                .collect();
            for edge in normalized {
                self.add_edge(edge);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::rk22_examples::sorted_transitions;
    use super::super::{DCW, NCW};
    use automata_core::upw;

    #[test]
    fn normalize_recolors_safe_transitions_between_safe_components() {
        // accepts iff only finitely many `a`s are read after the first `a`, i.e. all words with
        // finitely many `a`s. The safe components are {0} and {1}, which are connected by the
        // safe transition `⟨0, a, 1⟩`, so this is not normal.
        let mut dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
            ])
            .into_dcw(0);
        assert!(!dcw.is_normal());
        let words = [upw!("a"), upw!("b"), upw!("ab", "b"), upw!("b", "ab")];
        let accepted: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();

        dcw.normalize();
        assert!(dcw.is_normal());
        assert_eq!(
            sorted_transitions(&dcw),
            vec![
                (0, 'a', true, 1),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
            ]
        );
        let accepted_after: Vec<bool> = words.iter().map(|w| dcw.accepts(w)).collect();
        assert_eq!(accepted, accepted_after);
        assert_eq!(accepted, vec![false, true, true, false]);
    }

    #[test]
    fn normalize_keeps_normal_automaton() {
        // RK22, Figure 2: the safe components are {0, 1} and {2}, and all safe transitions
        // stay within them, so the automaton is already normal
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
        assert!(dcw.is_normal());
        let mut normalized = dcw.clone();
        normalized.normalize();
        assert_eq!(sorted_transitions(&normalized), sorted_transitions(&dcw));
    }

    #[test]
    fn normalize_merges_recolored_transition_with_existing_alpha_transition() {
        // `⟨0, a, 1⟩` exists both as a safe and as an `α`-transition. The safe one connects the
        // safe components {0} and {1, 2}, and recoloring it makes it coincide with the other.
        // The safe chain `1 → 2 → 1` forms a single safe component and must stay safe.
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 1),
                (0, 'a', true, 1),
                (0, 'b', true, 0),
                (1, 'a', false, 2),
                (1, 'b', true, 1),
                (2, 'a', false, 1),
                (2, 'b', true, 2),
            ])
            .into_ncw(0);
        assert!(ncw.is_safe_deterministic());
        assert!(!ncw.is_normal());
        ncw.normalize();
        assert!(ncw.is_normal());
        assert!(ncw.is_safe_deterministic());
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', true, 1),
                (0, 'b', true, 0),
                (1, 'a', false, 2),
                (1, 'b', true, 1),
                (2, 'a', false, 1),
                (2, 'b', true, 2),
            ]
        );
    }
}
