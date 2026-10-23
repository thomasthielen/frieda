//! Measures the running time of the minimization pipeline of AK22 (see `NCW::minimize`) on random
//! safe deterministic GFG-tNCWs, both for each step separately and for the complete pipeline.
//!
//! For every size, `SAMPLES` automata are drawn with `generate_random_ncw` (seeded, so runs are
//! reproducible). Each automaton is minimized once with `NCW::minimize_reporting`, which runs
//! exactly the steps of `NCW::minimize`, including the reuse of results across steps, and reports
//! the end of every step, so that each step and the complete pipeline can be timed in one run.
//! The reported times are medians over the samples.
//!
//! Run in release mode, optionally passing the sizes to measure:
//! cargo run --release --example minimization_benchmark -p automata -- 10 20 40

use automata::TransitionSystem;
use automata::automaton::{MinimizationStep as Step, NCW};
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

/// The steps of `NCW::minimize` in the order they are reported, each with its column title and
/// whether its time is reported. The trivially fast steps are not reported, but still counted in
/// the time of the complete pipeline.
const STEPS: [(Step, &str, bool); 8] = [
    (Step::SafeDeterminismCheck, "safe-det check", false),
    (Step::ContainmentGame, "containment game", true),
    (Step::SemanticDeterminization, "sem. determinize", true),
    (Step::UnreachableRemoval, "remove unreachable", false),
    (Step::Normalization, "normalize", false),
    (Step::LanguageClasses, "language classes", false),
    (Step::SafeCentralization, "safe centralize", true),
    (Step::SafeMinimization, "safe minimize", true),
];

/// Minimizes `ncw`, returning the time of each step (in the order of `STEPS`) and of the
/// complete pipeline.
fn time_minimization(ncw: &mut NCW) -> (Vec<Duration>, Duration) {
    let mut step_times = Vec::with_capacity(STEPS.len());
    let start = Instant::now();
    let mut step_start = start;
    ncw.minimize_reporting(|step| {
        let duration = step_start.elapsed();
        assert_eq!(step, STEPS[step_times.len()].0, "unexpected order of steps");
        step_times.push(duration);
        // excludes the bookkeeping above from the time of the next step
        step_start = Instant::now();
    });
    (step_times, start.elapsed())
}

/// Returns the number of transitions of `ncw`.
fn edge_count(ncw: &NCW) -> usize {
    ncw.state_indices()
        .map(|q| ncw.transitions_from(q).count())
        .sum()
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
        .chain(
            STEPS
                .into_iter()
                .filter(|&(_, _, reported)| reported)
                .map(|(_, title, _)| title),
        )
        .chain(["pipeline"])
        .map(String::from)
        .collect();
    // wide enough for the header and for values up to 7 digits
    let widths: Vec<usize> = header.iter().map(|title| title.len().max(7)).collect();
    print_row(&header, &widths);

    for size in sizes {
        let mut step_times: Vec<Vec<Duration>> = vec![vec![]; STEPS.len()];
        let mut pipeline_times = vec![];
        let (mut states, mut edges, mut minimal_states) = (0, 0, 0);

        for sample in 0..SAMPLES {
            let mut rng = StdRng::seed_from_u64(SEED + sample);
            let mut ncw =
                generate_random_ncw(SYMBOLS, size, ALPHA_PROBABILITY, NONDETERMINISM, &mut rng);
            states += ncw.size();
            edges += edge_count(&ncw);

            let (times, pipeline_time) = time_minimization(&mut ncw);
            for (step, duration) in times.into_iter().enumerate() {
                step_times[step].push(duration);
            }
            pipeline_times.push(pipeline_time);
            minimal_states += ncw.size();
        }

        let average = |total: usize| format!("{:.1}", total as f64 / SAMPLES as f64);
        let row: Vec<String> = [size.to_string(), average(states), average(edges)]
            .into_iter()
            .chain([average(minimal_states)])
            .chain(
                step_times
                    .into_iter()
                    .zip(STEPS)
                    .filter(|&(_, (_, _, reported))| reported)
                    .map(|(times, _)| millis(median(times))),
            )
            .chain([millis(median(pipeline_times))])
            .collect();
        print_row(&row, &widths);
    }
}
