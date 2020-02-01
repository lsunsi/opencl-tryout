#![feature(custom_test_frameworks)]
#![test_runner(criterion::runner)]

use criterion::{black_box, Criterion};
use criterion_macro::criterion;
use opencl_tryout::{cpu, gpu};

fn input() -> (Vec<f32>, (f32, f32)) {
    let wealths: Vec<_> = (1..1000).map(|a| a as f32).collect();
    let portfolio = (0.07, 0.11);
    (wealths, portfolio)
}

#[criterion]
fn bench_cpu(c: &mut Criterion) {
    let (wealths, portfolio) = input();
    c.bench_function("Bench-CPU", |b| {
        b.iter(|| cpu::calculate(black_box(&wealths), black_box(portfolio)))
    });
}

#[criterion]
fn bench_gpu(c: &mut Criterion) {
    let (wealths, portfolio) = input();
    c.bench_function("Bench-GPU", |b| {
        b.iter(|| gpu::calculate(black_box(&wealths), black_box(portfolio)))
    });
}
