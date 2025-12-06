pub mod analysis;
pub mod model2d;
pub mod model3d;
pub mod model4d;

pub use model2d::IsingModel2D;
pub use model3d::IsingModel3D;
pub use model4d::IsingModel4D;

#[cfg(test)]
mod tests;
