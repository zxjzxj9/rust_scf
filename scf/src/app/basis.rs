use crate::config::Config;
use crate::io::fetch_basis;
use ::basis::basis::AOBasis;
use ::basis::cgto::Basis631G;
use color_eyre::eyre::{eyre, Result};
use periodic_table_on_an_enum::Element;
use std::collections::HashMap;
use std::marker::PhantomData;
use tracing::info;

pub type BasisMap<B> = HashMap<&'static str, &'static B>;

/// Loader trait that can be implemented for any atomic-orbital basis family.
pub trait BasisLoader<B: AOBasis> {
    fn load(&self, element: &Element, requested: Option<&str>) -> Result<B>;
}

/// Default loader backed by the historical 6-31G basis helper.
pub struct Basis631GLoader;

impl BasisLoader<Basis631G> for Basis631GLoader {
    fn load(&self, element: &Element, requested: Option<&str>) -> Result<Basis631G> {
        if let Some(name) = requested {
            if name.to_lowercase() != "6-31g" {
                return Err(eyre!(
                    "Unsupported basis '{}' for element {}. Supply 6-31g or extend BasisLoader.",
                    name,
                    element.get_symbol()
                ));
            }
        }

        fetch_basis(element.get_symbol())
    }
}

/// Registry that caches loaded basis functions per element and makes it easy
/// to swap the underlying loader in the future.
pub struct BasisRegistry<B: AOBasis + 'static, L: BasisLoader<B>> {
    loader: L,
    cache: BasisMap<B>,
    _marker: PhantomData<B>,
}

impl<B: AOBasis + 'static, L: BasisLoader<B>> BasisRegistry<B, L> {
    pub fn new(loader: L) -> Self {
        Self {
            loader,
            cache: HashMap::new(),
            _marker: PhantomData,
        }
    }

    pub fn load_for_elements(
        &mut self,
        config: &Config,
        elements: &[Element],
    ) -> Result<BasisMap<B>> {
        for element in elements {
            let symbol = element.get_symbol();
            if self.cache.contains_key(symbol) {
                continue;
            }

            let requested = config.basis_sets.get(symbol).map(|s| s.as_str());
            if let Some(name) = requested {
                info!("Loading {} basis for {}", name, symbol);
            } else {
                info!("Loading default basis for {}", symbol);
            }

            let basis = self.loader.load(element, requested)?;
            let leaked: &'static B = Box::leak(Box::new(basis));
            self.cache.insert(symbol, leaked);
        }

        Ok(self.cache.clone())
    }
}
