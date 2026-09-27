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

    /// Removes all states of `self` that are not reachable from the initial state, together with
    /// all transitions leaving or entering them, and returns the removed states with their
    /// colors. The indices of the remaining states are unchanged.
    ///
    /// Following \[RK22, Theorem 2.2\], this is the step after [`Self::semantically_determinize`]:
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

    /// Returns, for every state of `self`, the index of its safe component. Following \[RK22\],
    /// the safe components of a tNCW are the SCCs of the graph obtained by removing all
    /// `α`-transitions (colored `true`, see [`DCW`] for the naming). Two states are in the same
    /// safe component iff they are mapped to the same index.
    fn safe_component_indices(&self) -> BTreeMap<D::StateIndex, usize> {
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
    /// for a [`DCW`]. Note that even for a deterministic `self`, `B_S` may be nondeterministic.
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
}

#[cfg(test)]
mod tests {
    use super::{DCW, NCW};
    use crate::ts::TSBuilder;
    use crate::{Pointed, TransitionSystem};
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

    /// RK22, Figure 2: a nice tDCW whose states are all equivalent, but differ in their safe
    /// languages. Its safe components are `{q0, q1}` and `{q2}`.
    const RK22_FIGURE_2: [(u32, char, bool, u32); 9] = [
        (0, 'a', false, 0),
        (0, 'b', false, 1),
        (0, 'c', true, 2),
        (1, 'a', true, 2),
        (1, 'b', true, 2),
        (1, 'c', false, 0),
        (2, 'a', false, 2),
        (2, 'b', true, 1),
        (2, 'c', true, 0),
    ];

    fn rk22_figure_2() -> DCW {
        DCW::builder().with_edges(RK22_FIGURE_2).into_dcw(0)
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

    /// The tNCW `B_S` for `S = {{q0, q1}}` of RK22, Figure 4, obtained from Figure 2.
    const RK22_FIGURE_4: [(u32, char, bool, u32); 9] = [
        (0, 'a', false, 0),
        (0, 'b', false, 1),
        (0, 'c', true, 0),
        (0, 'c', true, 1),
        (1, 'a', true, 0),
        (1, 'a', true, 1),
        (1, 'b', true, 0),
        (1, 'b', true, 1),
        (1, 'c', false, 0),
    ];

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
    fn dcw_into_ncw_keeps_states_transitions_and_initial_state() {
        let dcw = DCW::builder().with_edges(RK22_FIGURE_2).into_dcw(2);
        let ncw = dcw.clone().into_ncw();
        assert_eq!(ncw.initial(), 2);
        assert_eq!(
            ncw.state_indices().collect::<Vec<_>>(),
            dcw.state_indices().collect::<Vec<_>>()
        );
        assert_eq!(sorted_transitions(&ncw), sorted_transitions(&dcw));
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
}
