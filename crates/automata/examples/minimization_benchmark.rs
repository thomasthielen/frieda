//! Measures the running time of the minimization pipeline of AK22 (see `NCW::minimize`) on random
//! safe deterministic GFG-tNCWs, both for each step separately and for the complete pipeline.
//!
//! For every size, `SAMPLES` automata are drawn with `generate_random_ncw` (seeded, so runs are
//! reproducible). Each automaton is minimized twice: once step by step, timing every step, and
//! once with a single call of `NCW::minimize` on a fresh copy. Both results are checked to agree.
//! The reported times are medians over the samples.
//!
//! Run in release mode, optionally passing the sizes to measure:
//! cargo run --release --example minimization_benchmark -p automata -- 10 20 40

use automata::TransitionSystem;
use automata::automaton::NCW;
use automata::random::generate_random_ncw;
use rand::{SeedableRng, rngs::StdRng};
use std::time::{Duration, Instant};

/// The default sizes (number of states before removing unreachable ones) to measure.
const SIZES: [usize; 5] = [10, 20, 40, 80, 160];
/// The number of symbols of the alphabet.
const SYMBOLS: usize = 2;
/// The probability for a transition of the underlying DCW to be an `α`-transition.
const ALPHA_PROBABILITY: f64 = 0.3;
/// The probability for a transition of the underlying DCW to get an additional `α`-transition.
const NONDETERMINISM: f64 = 0.3;
/// The number of automata drawn per size.
const SAMPLES: u64 = 10;
/// The seed of the first sample; sample `i` uses `SEED + i`.
const SEED: u64 = 0;

const STEPS: [&str; 6] = [
    "safe-det check",
    "sem. determinize",
    "remove unreachable",
    "normalize",
    "safe centralize",
    "safe minimize",
];

/// Times `f` applied to `ncw`.
fn time(ncw: &mut NCW, f: impl FnOnce(&mut NCW)) -> Duration {
    let start = Instant::now();
    f(ncw);
    start.elapsed()
}

/// Returns the transitions of `ncw` as a sorted list, for comparing results.
fn sorted_transitions(ncw: &NCW) -> Vec<(u32, char, bool, u32)> {
    let mut transitions: Vec<_> = ncw
        .state_indices()
        .flat_map(|q| ncw.transitions_from(q).collect::<Vec<_>>())
        .collect();
    transitions.sort();
    transitions
}

fn median(mut durations: Vec<Duration>) -> Duration {
    durations.sort();
    durations[durations.len() / 2]
}

/// Prints `cells` as a table row, right-aligning each cell to the corresponding width.
fn print_row(cells: &[String], widths: &[usize]) {
    let cells: Vec<String> = cells
        .iter()
        .zip(widths)
        .map(|(cell, &width)| format!("{cell:>width$}"))
        .collect();
    println!("{}", cells.join(" | "));
}

fn millis(duration: Duration) -> String {
    format!("{:.3}", duration.as_secs_f64() * 1000.0)
}

fn main() {
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("sizes must be positive integers"))
        .collect();
    let sizes = if sizes.is_empty() {
        SIZES.to_vec()
    } else {
        sizes
    };

    println!(
        "{SYMBOLS} symbols, α-probability {ALPHA_PROBABILITY}, nondeterminism {NONDETERMINISM}, \
         {SAMPLES} samples per size, median times in ms"
    );
    let header: Vec<String> = ["size", "states", "edges", "min. states"]
        .into_iter()
        .chain(STEPS)
        .chain(["sum of steps", "pipeline"])
        .map(String::from)
        .collect();
    // wide enough for the header and for values up to 7 digits
    let widths: Vec<usize> = header.iter().map(|title| title.len().max(7)).collect();
    print_row(&header, &widths);

    for size in sizes {
        let mut step_times: Vec<Vec<Duration>> = vec![vec![]; STEPS.len()];
        let mut sum_times = vec![];
        let mut pipeline_times = vec![];
        let (mut states, mut edges, mut minimal_states) = (0, 0, 0);

        for sample in 0..SAMPLES {
            let mut rng = StdRng::seed_from_u64(SEED + sample);
            let input =
                generate_random_ncw(SYMBOLS, size, ALPHA_PROBABILITY, NONDETERMINISM, &mut rng);
            states += input.size();
            edges += sorted_transitions(&input).len();

            // step by step, in the order of `NCW::minimize`
            let mut ncw = input.clone();
            let times = [
                time(&mut ncw, |ncw| assert!(ncw.is_safe_deterministic())),
                time(&mut ncw, |ncw| ncw.semantically_determinize()),
                time(&mut ncw, |ncw| {
                    ncw.remove_unreachable_states();
                }),
                time(&mut ncw, |ncw| ncw.normalize()),
                time(&mut ncw, |ncw| ncw.safe_centralize()),
                time(&mut ncw, |ncw| ncw.safe_minimize()),
            ];
            for (step, duration) in times.into_iter().enumerate() {
                step_times[step].push(duration);
            }
            sum_times.push(times.iter().sum());
            minimal_states += ncw.size();

            // complete pipeline
            let mut minimized = input.clone();
            pipeline_times.push(time(&mut minimized, |ncw| ncw.minimize()));
            assert_eq!(
                sorted_transitions(&minimized),
                sorted_transitions(&ncw),
                "step-by-step and complete minimization disagree for size {size}, sample {sample}"
            );
        }

        let average = |total: usize| format!("{:.1}", total as f64 / SAMPLES as f64);
        let row: Vec<String> = [size.to_string(), average(states), average(edges)]
            .into_iter()
            .chain([average(minimal_states)])
            .chain(step_times.into_iter().map(|times| millis(median(times))))
            .chain([millis(median(sum_times)), millis(median(pipeline_times))])
            .collect();
        print_row(&row, &widths);
    }
}
