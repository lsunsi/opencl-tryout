use opencl_tryout::{cpu, gpu};

#[test]
fn implementations_match() {
    let wealths: Vec<_> = (1..100).map(|a| a as f32).collect();
    let portfolio = (0.07, 0.11);

    let pro_que = gpu::setup(wealths.len());

    let result_cpu = cpu::calculate(&wealths, portfolio);
    let result_gpu = gpu::calculate(&wealths, portfolio, &pro_que);

    assert_eq!(result_cpu.len(), result_gpu.len());

    for (rc, rg) in result_cpu.into_iter().zip(result_gpu) {
        assert!((rc - rg).abs() < 1e-6);
    }
}
