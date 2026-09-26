use crate::automaton::Semantics;
use crate::core::{
    Color,
    alphabet::{Alphabet, Expression},
};
use crate::games::{ParityGame, Player};
use crate::ts::{Deterministic, Shrinkable, Sproutable, StateColor};
use crate::{Automaton, DTS, NTS, TransitionSystem, automaton::InfiniteWordAutomaton, ts::run};
use automata_core::Void;
use automata_core::alphabet::CharAlphabet;
use std::collections::{BTreeMap, BTreeSet};

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

    /// Computes the language containment relation between the states of `self`, which is
    /// assumed to be a GFG-tNCW. The result contains the pair `(q, s)` iff
    /// `L(A^q) ⊆ L(A^s)`, where `A^q` is `self` with initial state `q`.
    ///
    /// Following \[KS15, Theorem 13\], containment is decided with the game `G(A^s, A^q)`,
    /// played on pairs of states `(p, r)` (starting in `(s, q)`). In every round
    /// 1. Spoiler picks a letter `σ`,
    /// 2. Duplicator picks a transition `⟨p, σ, p'⟩`,
    /// 3. Spoiler picks a transition `⟨r, σ, r'⟩`,
    ///
    /// and the game continues in `(p', r')`. Duplicator wins a play iff her run (from `s`) is
    /// accepting or Spoiler's run (from `q`) is rejecting, i.e. iff Spoiler's run takes
    /// `α`-transitions infinitely often whenever Duplicator's run does. This is a parity
    /// condition with three priorities, and all pairs are decided at once by solving a single
    /// [`ParityGame`]. To keep the arena total, missing transitions are treated as
    /// `α`-transitions into an implicit rejecting sink.
    ///
    /// If Duplicator wins, then `L(A^q) ⊆ L(A^s)` holds regardless of GFGness. The converse
    /// (and hence exactness of the relation) requires that `A^s` is GFG: Duplicator then wins
    /// by following the strategy that witnesses `A^s`'s GFGness. Note that a GFG-tNCW may
    /// contain states that are not GFG (e.g. states that are never used by the witnessing
    /// strategy); for pairs `(q, s)` with such an `s`, the returned relation may miss
    /// containments, but never contains a wrong one.
    pub fn gfg_containment_relation(&self) -> BTreeSet<(D::StateIndex, D::StateIndex)> {
        let states: Vec<D::StateIndex> = self.state_indices().collect();
        let letters: Vec<_> = self.alphabet().universe().collect();
        if letters.is_empty() {
            // there are no infinite words, so all languages are empty and hence equal
            return states
                .iter()
                .flat_map(|&q| states.iter().map(move |&s| (q, s)))
                .collect();
        }

        // states are numbered `0..n`, the implicit rejecting sink is `n`
        let n = states.len();
        let m = n + 1;
        let sink = n;
        let state_number: BTreeMap<D::StateIndex, usize> =
            states.iter().enumerate().map(|(i, &q)| (q, i)).collect();
        let letter_number: BTreeMap<_, usize> =
            letters.iter().enumerate().map(|(i, &a)| (a, i)).collect();

        // successors[p][a] contains all `(color, p')` with `⟨p, a, p'⟩` in δ (`color` is
        // `true` iff it is an `α`-transition), using the sink if there are none
        let mut successors = vec![vec![Vec::new(); letters.len()]; m];
        for (p, &q) in states.iter().enumerate() {
            for (_, sym, color, target) in self.transitions_from(q) {
                successors[p][letter_number[&sym]].push((color, state_number[&target]));
            }
        }
        for per_letter in successors.iter_mut() {
            for succs in per_letter.iter_mut() {
                if succs.is_empty() {
                    succs.push((true, sink));
                }
            }
        }

        // Vertices of the game (`p` is Duplicator's state, `r` Spoiler's):
        // - `choose_letter(p, r, k)`: Spoiler picks a letter. `k` is the priority of the round
        //   that led here: 2 if Spoiler took an `α`-transition, otherwise 1 if Duplicator took
        //   one, otherwise 0. Duplicator (Even) wins iff the greatest priority seen infinitely
        //   often is even, i.e. iff `Inf(α_p) → Inf(α_r)`.
        // - `duplicator_moves(p, r, a)`: Duplicator picks an `a`-successor of `p`.
        // - `spoiler_moves(p', c, r, a)`: Spoiler picks an `a`-successor of `r`, where `c` is
        //   the color of the transition Duplicator just took to reach `p'`.
        let l = letters.len();
        let choose_letter = |p: usize, r: usize, k: usize| (p * m + r) * 3 + k;
        let duplicator_base = m * m * 3;
        let duplicator_moves = |p: usize, r: usize, a: usize| duplicator_base + (p * m + r) * l + a;
        let spoiler_base = duplicator_base + m * m * l;
        let spoiler_moves = |p: usize, c: bool, r: usize, a: usize| {
            spoiler_base + ((p * 2 + c as usize) * m + r) * l + a
        };

        let mut game = ParityGame::new();
        for _ in 0..m * m {
            for k in 0..3 {
                game.add_vertex(Player::Odd, k);
            }
        }
        for _ in 0..m * m * l {
            game.add_vertex(Player::Even, 0);
        }
        for _ in 0..m * 2 * m * l {
            game.add_vertex(Player::Odd, 0);
        }

        for (p, per_letter) in successors.iter().enumerate() {
            for r in 0..m {
                for (a, succs) in per_letter.iter().enumerate() {
                    for k in 0..3 {
                        game.add_edge(choose_letter(p, r, k), duplicator_moves(p, r, a));
                    }
                    for &(c, p_next) in succs {
                        game.add_edge(duplicator_moves(p, r, a), spoiler_moves(p_next, c, r, a));
                    }
                }
            }
        }
        for p_next in 0..m {
            for c in [false, true] {
                for (r, per_letter) in successors.iter().enumerate() {
                    for (a, succs) in per_letter.iter().enumerate() {
                        for &(d, r_next) in succs {
                            let k = if d { 2 } else { c as usize };
                            game.add_edge(
                                spoiler_moves(p_next, c, r, a),
                                choose_letter(p_next, r_next, k),
                            );
                        }
                    }
                }
            }
        }

        let winner = game.solve();
        let mut relation = BTreeSet::new();
        for (qi, &q) in states.iter().enumerate() {
            for (si, &s) in states.iter().enumerate() {
                // Duplicator plays in `A^s`, Spoiler in `A^q`; the priority of the initial
                // vertex is irrelevant, as the winning condition is prefix-independent
                if winner[choose_letter(si, qi, 0)] == Player::Even {
                    relation.insert((q, s));
                }
            }
        }
        relation
    }

    /// Returns `true` iff `self`, which is assumed to be a GFG-tNCW, is semantically
    /// deterministic. Following \[RK22\], this is the case if different nondeterministic
    /// choices lead to equivalent states: for every state `q` and letter `σ`, and all
    /// transitions `⟨q, σ, s⟩` and `⟨q, σ, s'⟩` (of any color), we have `L(A^s) = L(A^s')`.
    ///
    /// Language equivalence is decided with [`Self::gfg_containment_relation`], so the result
    /// is exact if all `σ`-successors of all states are GFG. Otherwise, some equivalences might
    /// be missed, in which case `false` may be returned for a semantically deterministic
    /// automaton; a returned `true` is always correct.
    pub fn is_semantically_deterministic(&self) -> bool {
        let contained = self.gfg_containment_relation();
        let equivalent = |s, t| contained.contains(&(s, t)) && contained.contains(&(t, s));
        self.state_indices().all(|q| {
            let mut successors: BTreeMap<_, Vec<D::StateIndex>> = BTreeMap::new();
            for (_, sym, _, target) in self.transitions_from(q) {
                successors.entry(sym).or_default().push(target);
            }
            successors
                .values()
                .all(|targets| targets.iter().all(|&s| equivalent(targets[0], s)))
        })
    }

    /// Semantically determinizes `self`, which is assumed to be a GFG-tNCW, by removing all
    /// transitions that are not covering. Following \[RK22\], a transition `⟨q, σ, s⟩` is
    /// *covering* if for every transition `⟨q, σ, s'⟩`, it holds that `L(A^s') ⊆ L(A^s)`.
    ///
    /// Since the transitions used by a strategy witnessing GFGness are covering \[KS15\], this
    /// neither changes the language nor the GFGness of `self`, and afterwards all remaining
    /// `σ`-successors of a state are equivalent, i.e. [`Self::is_semantically_deterministic`]
    /// holds. Removing transitions also preserves safe determinism.
    ///
    /// Covering is decided with [`Self::gfg_containment_relation`], so this requires that all
    /// states of `self` are GFG (which is why \[RK22, Theorem 2.2\] removes non-GFG states
    /// first). Otherwise, containments might be missed and covering transitions removed.
    ///
    /// Edges whose expression matches several symbols are kept as they are if they are
    /// covering for all of them, and are otherwise replaced by one edge for each symbol they
    /// are covering for.
    pub fn semantically_determinize(&mut self)
    where
        D: Shrinkable + Sproutable,
    {
        let contained = self.gfg_containment_relation();
        let states: Vec<D::StateIndex> = self.state_indices().collect();
        for q in states {
            let mut successors: BTreeMap<_, BTreeSet<D::StateIndex>> = BTreeMap::new();
            for (_, sym, _, target) in self.transitions_from(q) {
                successors.entry(sym).or_default().insert(target);
            }
            let covering = |sym: &A::Symbol, s: D::StateIndex| {
                successors[sym].iter().all(|&t| contained.contains(&(t, s)))
            };

            let edges = self
                .remove_edges_from(q)
                .expect("state was obtained from state_indices");
            for (source, expression, color, target) in edges {
                let symbols: Vec<A::Symbol> = expression.symbols().collect();
                if symbols.iter().all(|sym| covering(sym, target)) {
                    self.add_edge((source, expression, color, target));
                    continue;
                }
                for sym in symbols.into_iter().filter(|sym| covering(sym, target)) {
                    let expression = self.alphabet().make_expression(sym);
                    self.add_edge((source, expression, color, target));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DCW, NCW};
    use crate::TransitionSystem;
    use crate::ts::TSBuilder;
    use automata_core::alphabet::CharAlphabet;
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

    #[test]
    fn gfg_containment_relation_of_simple_languages() {
        // state 0: universal (safe loops), state 1: empty (`α`-loops),
        // state 2: finitely many `a`s, state 3: empty, as it only has an `α`-loop on `a` and
        // no `b`-transition at all (exercising the implicit rejecting sink)
        let ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', true, 1),
                (2, 'a', true, 2),
                (2, 'b', false, 2),
                (3, 'a', true, 3),
            ])
            .into_ncw(0);
        let contained = ncw.gfg_containment_relation();
        for q in 0..4 {
            // every language contains itself and the empty languages
            assert!(contained.contains(&(q, q)));
            assert!(contained.contains(&(1, q)));
            assert!(contained.contains(&(3, q)));
            // the universal language contains everything
            assert!(contained.contains(&(q, 0)));
        }
        assert!(contained.contains(&(1, 3)) && contained.contains(&(3, 1)));
        assert!(!contained.contains(&(0, 2)));
        assert!(!contained.contains(&(0, 1)));
        assert!(!contained.contains(&(2, 1)));
        assert!(!contained.contains(&(2, 3)));
    }

    #[test]
    fn rk22_figure_2_states_are_equivalent() {
        // RK22, Example 3.1: all states of the tDCW in Figure 2 are equivalent
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
        let contained = dcw.gfg_containment_relation();
        assert_eq!(contained.len(), 9);
        assert!(dcw.is_semantically_deterministic());
    }

    #[test]
    fn rk22_figure_4_is_semantically_deterministic() {
        // RK22, Figure 4: the tNCW `B_S` for `S = {{q0, q1}}`, whose nondeterminism is on
        // `α`-transitions only, and whose states are all equivalent
        let ncw = NCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', false, 1),
                (1, 'c', false, 0),
                (0, 'c', true, 0),
                (0, 'c', true, 1),
                (1, 'a', true, 1),
                (1, 'b', true, 1),
                (1, 'a', true, 0),
                (1, 'b', true, 0),
            ])
            .into_ncw(0);
        assert_eq!(ncw.gfg_containment_relation().len(), 4);
        assert!(ncw.is_semantically_deterministic());
    }

    #[test]
    fn ncw_not_semantically_deterministic() {
        // state 0 has two `a`-successors: state 1 (universal) via a safe transition and
        // state 2 (empty) via an `α`-transition. The automaton is GFG (always move to 1) and
        // safe deterministic, but not semantically deterministic.
        let ncw = NCW::builder()
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
        assert!(ncw.is_safe_deterministic());
        assert!(!ncw.is_semantically_deterministic());
    }

    #[test]
    fn ncw_semantically_deterministic_with_structurally_different_successors() {
        // state 0 has two `a`-successors, 1 and 2, which both recognize "finitely many `a`s"
        // but in structurally different ways: 1 is a single state, while 2 and 3 alternate.
        let ncw = NCW::builder()
            .with_edges([
                (0, 'a', true, 1),
                (0, 'a', true, 2),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
                (2, 'a', true, 3),
                (2, 'b', false, 2),
                (3, 'a', true, 2),
                (3, 'b', false, 3),
            ])
            .into_ncw(0);
        assert!(ncw.is_semantically_deterministic());
    }

    #[test]
    fn dcw_is_semantically_deterministic() {
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(0);
        assert!(dcw.is_semantically_deterministic());
    }

    /// Returns the transitions of `ncw` as a sorted list, for comparing against expectations.
    fn sorted_transitions<
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

    #[test]
    fn semantically_determinize_removes_non_covering_transition() {
        // same automaton as in `ncw_not_semantically_deterministic`: the `α`-transition
        // `⟨0, a, 2⟩` leads to the empty state 2 and is not covering, as `L(1) ⊈ L(2)`
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
        assert_eq!(
            sorted_transitions(&ncw),
            vec![
                (0, 'a', false, 1),
                (0, 'b', false, 0),
                (1, 'a', false, 1),
                (1, 'b', false, 1),
                (2, 'a', true, 2),
                (2, 'b', true, 2),
            ]
        );
        assert!(ncw.is_safe_deterministic());
        assert!(ncw.is_semantically_deterministic());
    }

    #[test]
    fn semantically_determinize_keeps_greatest_successor() {
        // state 0 has three `a`-successors: 1 (finitely many `a`s), 2 (universal) and 3
        // (empty). Only the transition to 2 is covering, as `L(3) ⊊ L(1) ⊊ L(2)`.
        let mut ncw = NCW::builder()
            .with_edges([
                (0, 'a', true, 1),
                (0, 'a', true, 2),
                (0, 'a', false, 3),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
                (2, 'a', false, 2),
                (2, 'b', false, 2),
                (3, 'a', true, 3),
                (3, 'b', true, 3),
            ])
            .into_ncw(0);
        ncw.semantically_determinize();
        let from_0: Vec<_> = ncw.transitions_from(0).collect();
        assert_eq!(from_0.len(), 2);
        assert!(from_0.contains(&(0, 'a', true, 2)));
        assert!(from_0.contains(&(0, 'b', false, 0)));
        assert!(ncw.is_semantically_deterministic());
    }

    #[test]
    fn semantically_determinize_keeps_equivalent_successors() {
        // `rk22_figure_4_is_semantically_deterministic` and
        // `ncw_semantically_deterministic_with_structurally_different_successors`: all
        // nondeterministic choices lead to equivalent states, so every transition is covering
        let figure_4 = NCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', false, 1),
                (1, 'c', false, 0),
                (0, 'c', true, 0),
                (0, 'c', true, 1),
                (1, 'a', true, 1),
                (1, 'b', true, 1),
                (1, 'a', true, 0),
                (1, 'b', true, 0),
            ])
            .into_ncw(0);
        let structurally_different = NCW::builder()
            .with_edges([
                (0, 'a', true, 1),
                (0, 'a', true, 2),
                (0, 'b', false, 0),
                (1, 'a', true, 1),
                (1, 'b', false, 1),
                (2, 'a', true, 3),
                (2, 'b', false, 2),
                (3, 'a', true, 2),
                (3, 'b', false, 3),
            ])
            .into_ncw(0);
        for ncw in [figure_4, structurally_different] {
            let mut determinized = ncw.clone();
            determinized.semantically_determinize();
            assert_eq!(sorted_transitions(&determinized), sorted_transitions(&ncw));
        }
    }

    #[test]
    fn semantically_determinize_dcw_is_identity() {
        let dcw = DCW::builder()
            .with_edges([
                (0, 'a', false, 0),
                (0, 'b', true, 1),
                (1, 'a', false, 1),
                (1, 'b', true, 0),
            ])
            .into_dcw(0);
        let mut determinized = dcw.clone();
        determinized.semantically_determinize();
        assert_eq!(sorted_transitions(&determinized), sorted_transitions(&dcw));
    }
}
