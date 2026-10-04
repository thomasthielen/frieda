//! Two-player games on finite graphs.
//!
//! Currently, this only contains a solver for (max-)parity games, which is used to decide
//! language containment between states of a GFG-tNCW, see `gfg_containment_relation` on
//! [`crate::automaton::NCW`].

/// One of the two players of a [`ParityGame`]. [`Player::Even`] wins a play iff the greatest
/// priority that occurs infinitely often is even, [`Player::Odd`] wins otherwise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Player {
    /// The player who wins plays whose greatest infinitely-often priority is even.
    Even,
    /// The player who wins plays whose greatest infinitely-often priority is odd.
    Odd,
}

impl Player {
    /// Returns the player who wins if `priority` is the greatest one seen infinitely often.
    pub fn of_priority(priority: usize) -> Self {
        if priority.is_multiple_of(2) {
            Player::Even
        } else {
            Player::Odd
        }
    }

    /// Returns the other player.
    pub fn opponent(self) -> Self {
        match self {
            Player::Even => Player::Odd,
            Player::Odd => Player::Even,
        }
    }
}

/// A (max-)parity game on a finite arena. Vertices are identified by `usize` indices in
/// `0..self.size()`, each vertex is owned by a [`Player`] and carries a priority.
///
/// The arena must be *total*, i.e. every vertex needs at least one successor. This is not
/// checked when adding edges, but [`ParityGame::solve`] panics otherwise.
#[derive(Debug, Clone, Default)]
pub struct ParityGame {
    owner: Vec<Player>,
    priority: Vec<usize>,
    successors: Vec<Vec<usize>>,
    predecessors: Vec<Vec<usize>>,
}

impl ParityGame {
    /// Creates an empty game.
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of vertices.
    pub fn size(&self) -> usize {
        self.owner.len()
    }

    /// Adds a vertex owned by `owner` with the given `priority` and returns its index.
    pub fn add_vertex(&mut self, owner: Player, priority: usize) -> usize {
        self.owner.push(owner);
        self.priority.push(priority);
        self.successors.push(Vec::new());
        self.predecessors.push(Vec::new());
        self.owner.len() - 1
    }

    /// Adds an edge from `source` to `target`.
    pub fn add_edge(&mut self, source: usize, target: usize) {
        self.successors[source].push(target);
        self.predecessors[target].push(source);
    }

    /// Solves the game using Zielonka's recursive algorithm and returns, for every vertex, the
    /// [`Player`] who wins from it. Panics if some vertex has no successor.
    pub fn solve(&self) -> Vec<Player> {
        assert!(
            self.successors.iter().all(|s| !s.is_empty()),
            "parity game arena must be total"
        );
        let mut winner = vec![Player::Even; self.size()];
        let all = vec![true; self.size()];
        let (_, odd) = self.zielonka(&all);
        for (v, &is_odd) in odd.iter().enumerate() {
            if is_odd {
                winner[v] = Player::Odd;
            }
        }
        winner
    }

    /// Solves the subgame induced by the vertices in `arena` (which must be a trap for both
    /// players, so that the subgame is again total) and returns the winning regions of
    /// [`Player::Even`] and [`Player::Odd`], in this order.
    fn zielonka(&self, arena: &[bool]) -> (Vec<bool>, Vec<bool>) {
        let n = self.size();
        let Some(max_priority) = (0..n).filter(|&v| arena[v]).map(|v| self.priority[v]).max()
        else {
            return (vec![false; n], vec![false; n]);
        };
        let player = Player::of_priority(max_priority);
        let target: Vec<bool> = (0..n)
            .map(|v| arena[v] && self.priority[v] == max_priority)
            .collect();
        let attracted = self.attractor(arena, &target, player);
        let (sub_even, sub_odd) = self.zielonka(&difference(arena, &attracted));
        let sub_opponent = match player {
            Player::Even => sub_odd,
            Player::Odd => sub_even,
        };

        let (win_player, win_opponent) = if sub_opponent.iter().all(|&b| !b) {
            (arena.to_vec(), vec![false; n])
        } else {
            let attracted = self.attractor(arena, &sub_opponent, player.opponent());
            let (sub_even, sub_odd) = self.zielonka(&difference(arena, &attracted));
            let (sub_player, sub_opponent) = match player {
                Player::Even => (sub_even, sub_odd),
                Player::Odd => (sub_odd, sub_even),
            };
            (sub_player, union(&sub_opponent, &attracted))
        };

        match player {
            Player::Even => (win_player, win_opponent),
            Player::Odd => (win_opponent, win_player),
        }
    }

    /// Computes the attractor of `target` for `player` within the subgame induced by `arena`,
    /// i.e. the set of vertices from which `player` can force a visit to `target`.
    fn attractor(&self, arena: &[bool], target: &[bool], player: Player) -> Vec<bool> {
        let n = self.size();
        let mut attracted = target.to_vec();
        // for vertices of the opponent: number of successors (within `arena`) that are not yet
        // attracted; such a vertex is attracted once this drops to zero
        let mut remaining: Vec<usize> = (0..n)
            .map(|v| self.successors[v].iter().filter(|&&w| arena[w]).count())
            .collect();
        let mut queue: Vec<usize> = (0..n).filter(|&v| attracted[v]).collect();
        while let Some(w) = queue.pop() {
            for &v in &self.predecessors[w] {
                if !arena[v] || attracted[v] {
                    continue;
                }
                if self.owner[v] == player {
                    attracted[v] = true;
                    queue.push(v);
                } else {
                    remaining[v] -= 1;
                    if remaining[v] == 0 {
                        attracted[v] = true;
                        queue.push(v);
                    }
                }
            }
        }
        attracted
    }
}

// for each pair of a and b: returns true only on a = true and b = false
fn difference(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(&x, &y)| x && !y).collect()
}

fn union(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(&x, &y)| x || y).collect()
}

#[cfg(test)]
mod tests {
    use super::{ParityGame, Player};

    #[test]
    fn single_self_loops() {
        let mut game = ParityGame::new();
        let even = game.add_vertex(Player::Odd, 2);
        let odd = game.add_vertex(Player::Even, 1);
        game.add_edge(even, even);
        game.add_edge(odd, odd);
        assert_eq!(game.solve(), vec![Player::Even, Player::Odd]);
    }

    #[test]
    fn choices_matter() {
        // v0 (Even, 0) can move to v1 (priority 1, self-loop) or v2 (priority 2, self-loop).
        // v3 (Odd, 0) can move to v1 or v2 as well.
        let mut game = ParityGame::new();
        let v0 = game.add_vertex(Player::Even, 0);
        let v1 = game.add_vertex(Player::Even, 1);
        let v2 = game.add_vertex(Player::Even, 2);
        let v3 = game.add_vertex(Player::Odd, 0);
        for v in [v0, v3] {
            game.add_edge(v, v1);
            game.add_edge(v, v2);
        }
        game.add_edge(v1, v1);
        game.add_edge(v2, v2);
        assert_eq!(
            game.solve(),
            vec![Player::Even, Player::Odd, Player::Even, Player::Odd]
        );
    }

    #[test]
    fn cycle_through_high_priority() {
        // Odd owns v0 and can either stay in v0 (priority 1) forever or move to v1, which
        // has priority 2 and leads back to v0. Staying is winning for Odd.
        let mut game = ParityGame::new();
        let v0 = game.add_vertex(Player::Odd, 1);
        let v1 = game.add_vertex(Player::Even, 2);
        game.add_edge(v0, v0);
        game.add_edge(v0, v1);
        game.add_edge(v1, v0);
        assert_eq!(game.solve(), vec![Player::Odd, Player::Odd]);

        // if Even owns v0, she moves to v1 and sees priority 2 infinitely often
        let mut game = ParityGame::new();
        let v0 = game.add_vertex(Player::Even, 1);
        let v1 = game.add_vertex(Player::Even, 2);
        game.add_edge(v0, v0);
        game.add_edge(v0, v1);
        game.add_edge(v1, v0);
        assert_eq!(game.solve(), vec![Player::Even, Player::Even]);
    }

    #[test]
    fn icag_example() {
        // Example taken from the lecture "Infinite Computation and Games" by Christof Löding (RWTH)
        let mut game = ParityGame::new();
        let v0 = game.add_vertex(Player::Even, 3);
        let v1 = game.add_vertex(Player::Odd, 2);
        let v2 = game.add_vertex(Player::Odd, 2);
        let v3 = game.add_vertex(Player::Odd, 3);
        let v4 = game.add_vertex(Player::Odd, 2);
        let v5 = game.add_vertex(Player::Even, 0);
        let v6 = game.add_vertex(Player::Odd, 4);
        let v7 = game.add_vertex(Player::Even, 1);
        game.add_edge(v0, v1);
        game.add_edge(v0, v4);
        game.add_edge(v1, v0);
        game.add_edge(v1, v2);
        game.add_edge(v1, v5);
        game.add_edge(v2, v3);
        game.add_edge(v2, v5);
        game.add_edge(v3, v7);
        game.add_edge(v4, v0);
        game.add_edge(v4, v5);
        game.add_edge(v5, v1);
        game.add_edge(v5, v2);
        game.add_edge(v6, v2);
        game.add_edge(v6, v7);
        game.add_edge(v7, v6);
        game.add_edge(v7, v3);
        assert_eq!(
            game.solve(),
            vec![
                Player::Odd,
                Player::Odd,
                Player::Even,
                Player::Even,
                Player::Odd,
                Player::Even,
                Player::Even,
                Player::Even
            ]
        );
    }
}
