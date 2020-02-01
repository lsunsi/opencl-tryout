pub fn calculate(wealths: &[f32], (ret, vol): (f32, f32)) -> Vec<f32> {
    let mut result = vec![];

    for prev_wealth in wealths {
        for next_wealth in wealths {
            let a = vol.recip() * ((next_wealth / prev_wealth).ln() - (ret - vol.powi(2) / 2.0));
            let prob = (2.0 * std::f32::consts::PI).sqrt().recip() * (-a.powi(2) / 2.0).exp();
            result.push(prob);
        }
    }

    result
}
