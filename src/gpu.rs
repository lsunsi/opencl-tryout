use ocl::enums::PlatformInfo;
use ocl::{Device, Platform, ProQue};

const SRC: &str = r#"
    __kernel void add(__constant float* wealths, __global float* buffer, float ret, float vol) {
        uint idx = get_global_id(0);
        float prev_wealth = wealths[idx / 99];
        float next_wealth = wealths[idx % 99];
        float a = (log(next_wealth / prev_wealth) - (ret - vol * vol / 2.0)) / vol;
        buffer[idx] = 1.0 / sqrt(2.0 * M_PI) * exp(-a * a / 2.0);
    }
"#;

pub fn calculate(wealths: &[f32], (ret, vol): (f32, f32)) -> Vec<f32> {
    let platforms = Platform::list();
    assert!(platforms.len() == 1);
    let devices = Device::list_all(platforms[0]).unwrap();
    for device in devices {
        println!("{:?}", device.name());
    }

    let pro_que = ProQue::builder()
        .src(SRC)
        .dims(wealths.len() * wealths.len())
        .build()
        .unwrap();

    let wealths_buffer = pro_que.create_buffer::<f32>().unwrap();
    wealths_buffer.write(wealths).enq().unwrap();

    let probs_buffer = pro_que.create_buffer::<f32>().unwrap();

    let kernel = pro_que
        .kernel_builder("add")
        .arg(&wealths_buffer)
        .arg(&probs_buffer)
        .arg(ret)
        .arg(vol)
        .build()
        .unwrap();

    unsafe {
        kernel.enq().unwrap();
    }

    let mut vec = vec![0.0f32; probs_buffer.len()];
    probs_buffer.read(&mut vec).enq().unwrap();

    vec
}
