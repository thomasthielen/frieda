//! Language containment between states and semantic determinization \[AK22, Theorem 2.2\].
use super::CoBuchiCondition;
use crate::core::{
    Color,
    alphabet::{Alphabet, Expression},
};
use crate::games::{ParityGame, Player};
use crate::ts::{Shrinkable, Sproutable};
use crate::{Automaton, TransitionSystem};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::hash::Hash;

impl<A, Q, D> Automaton<A, CoBuchiCondition, Q, bool, D, true, true>
where
    A: Alphabet,
    Q: Color,
    D: TransitionSystem<Alphabet = A, StateColor = Q, EdgeColor = bool>,
{
    /// Computes the language containment relation between the states of `self`, which is
    /// assumed to be a GFG-tNCW.
    /// The result contains the pair `(q, s)` iff `L(A^q) ⊆ L(A^s)`,
    /// where `A^q` is `self` with initial state `q`.
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
    /// condition with three priorities (see `Vertex`), and all pairs are decided at once by
    /// solving a single [`ParityGame`]. To keep the arena total, missing transitions are
    /// treated as `α`-transitions into an implicit rejecting sink.
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

        // all `(is_α, p')` with `⟨p, a, p'⟩` in δ, where `None` is the implicit rejecting sink
        let mut transitions: HashMap<(Option<D::StateIndex>, A::Symbol), Vec<_>> = HashMap::new();
        for &q in &states {
            for (_, sym, color, target) in self.transitions_from(q) {
                transitions
                    .entry((Some(q), sym))
                    .or_default()
                    .push((color, Some(target)));
            }
        }
        let into_sink = [(true, None)];
        let successors = |p, a| {
            transitions
                .get(&(p, a))
                .map_or(&into_sink[..], Vec::as_slice)
        };

        // Duplicator plays in `A^s`, Spoiler in `A^q` (as we check for `L(A^q) ⊆ L(A^s)`).
        // Whether the initial vertex has priority 0 or 2 (spo_alpha = true) is irrelevant,
        // as the winning condition is prefix-independent
        let initial = |q, s| Vertex::ChooseLetter {
            dup: Some(s),
            spo: Some(q),
            spo_alpha: false,
        };

        let mut arena = Arena::new();
        for &q in &states {
            for &s in &states {
                arena.vertex(initial(q, s));
            }
        }
        while let Some((v, source)) = arena.unexplored.pop() {
            match v {
                Vertex::ChooseLetter { dup, spo, .. } => {
                    for &letter in &letters {
                        arena.edge(source, Vertex::DuplicatorMoves { dup, spo, letter });
                    }
                }
                Vertex::DuplicatorMoves { dup, spo, letter } => {
                    for &(dup_alpha, dup) in successors(dup, letter) {
                        let target = Vertex::SpoilerMoves {
                            dup,
                            dup_alpha,
                            spo,
                            letter,
                        };
                        arena.edge(source, target);
                    }
                }
                Vertex::SpoilerMoves {
                    dup, spo, letter, ..
                } => {
                    for &(spo_alpha, spo) in successors(spo, letter) {
                        let target = Vertex::ChooseLetter {
                            dup,
                            spo,
                            spo_alpha,
                        };
                        arena.edge(source, target);
                    }
                }
            }
        }

        let winner = arena.game.solve();
        let mut relation = BTreeSet::new();
        for &q in &states {
            for &s in &states {
                if winner[arena.index[&initial(q, s)]] == Player::Even {
                    relation.insert((q, s));
                }
            }
        }
        relation
    }

    /// Returns `true` iff `self`, which is assumed to be a GFG-tNCW, is semantically
    /// deterministic. Following \[AK22\], this is the case if different nondeterministic
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
    /// transitions that are not covering. Following \[AK22\], a transition `⟨q, σ, s⟩` is
    /// *covering* if for every transition `⟨q, σ, s'⟩`, it holds that `L(A^s') ⊆ L(A^s)`.
    ///
    /// Since the transitions used by a strategy witnessing GFGness are covering \[KS15\], this
    /// neither changes the language nor the GFGness of `self`, and afterwards all remaining
    /// `σ`-successors of a state are equivalent, i.e. [`Self::is_semantically_deterministic`]
    /// holds. Removing transitions also preserves safe determinism.
    ///
    /// Covering is decided with [`Self::gfg_containment_relation`], so this requires that all
    /// states of `self` are GFG (which is why \[AK22, Theorem 2.2\] removes non-GFG states
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

/// A vertex of the containment game `G(A^s, A^q)`, where `dup` and `spo` are the current states
/// of Duplicator and Spoiler, respectively, and `None` is the implicit rejecting sink.
///
/// A round visits [`Vertex::ChooseLetter`], [`Vertex::DuplicatorMoves`] and
/// [`Vertex::SpoilerMoves`] in this order. Its greatest priority is 2 if Spoiler took an
/// `α`-transition, otherwise 1 if Duplicator took one, and otherwise 0. Hence, Duplicator
/// ([`Player::Even`]) wins iff Spoiler takes `α`-transitions infinitely often whenever
/// Duplicator does.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Vertex<S, L> {
    /// Spoiler picks a letter. `spo_alpha` is `true` iff Spoiler's last transition was an
    /// `α`-transition, which gives this vertex priority 2.
    ChooseLetter {
        dup: Option<S>,
        spo: Option<S>,
        spo_alpha: bool,
    },
    /// Duplicator picks a `letter`-transition from `dup`.
    DuplicatorMoves {
        dup: Option<S>,
        spo: Option<S>,
        letter: L,
    },
    /// Spoiler picks a `letter`-transition from `spo`. `dup_alpha` is `true` iff the transition
    /// that Duplicator just took to `dup` is an `α`-transition, which gives this vertex
    /// priority 1.
    SpoilerMoves {
        dup: Option<S>,
        dup_alpha: bool,
        spo: Option<S>,
        letter: L,
    },
}

impl<S, L> Vertex<S, L> {
    fn owner(&self) -> Player {
        match self {
            Vertex::DuplicatorMoves { .. } => Player::Even,
            Vertex::ChooseLetter { .. } | Vertex::SpoilerMoves { .. } => Player::Odd,
        }
    }
    fn priority(&self) -> usize {
        match *self {
            Vertex::ChooseLetter { spo_alpha, .. } => 2 * spo_alpha as usize,
            Vertex::DuplicatorMoves { .. } => 0,
            Vertex::SpoilerMoves { dup_alpha, .. } => dup_alpha as usize,
        }
    }
}

/// A [`ParityGame`] whose arena is built on the fly, starting from the initial vertices.
///
/// Every [`Vertex`] is added to `game` the first time it is encountered, with `index` mapping
/// it to its index in `game`. It is also pushed onto the stack `unexplored`, which will be
/// popped until empty, adding the outgoing edges of each popped vertex (and thereby discovering
/// new vertices), so that afterwards all reachable vertices and their edges are in `game`.
struct Arena<S, L> {
    game: ParityGame,
    index: HashMap<Vertex<S, L>, usize>,
    unexplored: Vec<(Vertex<S, L>, usize)>,
}

impl<S: Copy + Eq + Hash, L: Copy + Eq + Hash> Arena<S, L> {
    fn new() -> Self {
        Self {
            game: ParityGame::new(),
            index: HashMap::new(),
            unexplored: Vec::new(),
        }
    }

    /// Returns the index of `v` in the game, adding it first if necessary.
    fn vertex(&mut self, v: Vertex<S, L>) -> usize {
        if let Some(&i) = self.index.get(&v) {
            return i;
        }
        let i = self.game.add_vertex(v.owner(), v.priority());
        self.index.insert(v, i);
        self.unexplored.push((v, i));
        i
    }

    /// Adds an edge from the vertex with index `source` to `target`.
    fn edge(&mut self, source: usize, target: Vertex<S, L>) {
        let target = self.vertex(target);
        self.game.add_edge(source, target);
    }
}

#[cfg(test)]
mod tests {
    use super::super::ak22_examples::sorted_transitions;
    use super::super::{DCW, NCW};
    use crate::TransitionSystem;

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
    fn ak22_figure_2_states_are_equivalent() {
        // AK22, Example 3.1: all states of the tDCW in Figure 2 are equivalent
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
    fn ak22_figure_4_is_semantically_deterministic() {
        // AK22, Figure 4: the tNCW `B_S` for `S = {{q0, q1}}`, whose nondeterminism is on
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
        // `ak22_figure_4_is_semantically_deterministic` and
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
