use nalgebra::Vector3;

// Simpson's rule integration
pub(crate) fn simpson_integration<F>(f: F, a: f64, b: f64, n: usize) -> f64
where
    F: Fn(f64) -> f64,
{
    let n = if n % 2 == 0 { n } else { n + 1 };
    let h = (b - a) / n as f64;

    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 * f(x) } else { 4.0 * f(x) };
    }
    sum * h / 3.0
}


// Helper function to determine Simpson's weight for a given index and max index
fn simpson_weight(i: usize, n: usize) -> f64 {
    if i == 0 || i == n {
        1.0
    } else if i % 2 == 1 {
        4.0
    } else {
        2.0
    }
}

pub(crate) fn simpson_integration_3d<F>(
    f: F,
    a: Vector3<f64>,
    b: Vector3<f64>,
    nx: usize,
    ny: usize,
    nz: usize
) -> f64
where
    F: Fn(f64, f64, f64) -> f64,
{
    // Ensure nx, ny, nz are even
    let nx = if nx % 2 == 0 { nx } else { nx + 1 };
    let ny = if ny % 2 == 0 { ny } else { ny + 1 };
    let nz = if nz % 2 == 0 { nz } else { nz + 1 };

    let hx = (b.x - a.x) / nx as f64;
    let hy = (b.y - a.y) / ny as f64;
    let hz = (b.z - a.z) / nz as f64;

    let mut sum = 0.0;

    for i in 0..=nx {
        let x = a.x + i as f64 * hx;
        let wx = simpson_weight(i, nx);
        for j in 0..=ny {
            let y = a.y + j as f64 * hy;
            let wy = simpson_weight(j, ny);
            for k in 0..=nz {
                let z = a.z + k as f64 * hz;
                let wz = simpson_weight(k, nz);
                let w = wx * wy * wz;
                sum += w * f(x, y, z);
            }
        }
    }

    // Multiply by the step sizes and the factor 1/27
    sum * (hx * hy * hz) / 27.0
}