use std::fs::File;
use std::io::{Read, Write};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_pickle;
use serde_pickle::Serializer;
use crate::gto::GTO;

// need to consider how to reuse GTO integral, since s, p share the same exponents
#[derive(Debug, Serialize, Deserialize)]
pub struct ContractedGTO {
    pub primitives: Vec<GTO>,
    pub coefficients: Vec<f64>,
    // shell_type: 1s, 2s, 2px, 2py, 2pz, ...
    pub shell_type: String,
    pub n: i32,
    pub l: i32,
    pub m: i32,
    pub s: i32, // +1 or -1, stand for alpha or beta
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Basis631G {
    // define of the basis set
    pub name: String,
    // define atomic number
    pub atomic_number: i32,
    pub basis_set: Vec<ContractedGTO>,
}

enum BasisFormat {
    NWChem,
    Json,
}

/// Possible shell types.
#[derive(Debug)]
pub enum ShellType {
    S,
    P,
    D,
    SP, // Combined S/P in one block, as in NWChem
    // etc.
}

/// Helper to parse shell type from a string like "S" or "SP"
fn parse_shell_type(s: &str) -> Option<ShellType> {
    match s.to_ascii_uppercase().as_str() {
        "S" => Some(ShellType::S),
        "P" => Some(ShellType::P),
        "D" => Some(ShellType::D),
        "SP" => Some(ShellType::SP),
        // ...
        _ => None,
    }
}

/// Helper to parse floats in NWChem style (0.1172280000E+05, etc.)
fn parse_nwchem_float(s: &str) -> Result<f64, std::num::ParseFloatError> {
    // In many cases, Rust's default `f64::from_str` can handle the typical
    // "1.2340E+02" format. If not, you can do extra transformations here.
    s.parse::<f64>()
}

impl Basis631G {

    // Example of nwchem format:
    // #----------------------------------------------------------------------
    // # Basis Set Exchange
    // # Version 0.10
    // # https://www.basissetexchange.org
    // #----------------------------------------------------------------------
    // #   Basis set: 6-31G
    // # Description: 6-31G valence double-zeta
    // #        Role: orbital
    // #     Version: 1  (Data from Gaussian 09/GAMESS)
    // #----------------------------------------------------------------------
    //
    //
    // BASIS "ao basis" SPHERICAL PRINT
    // #BASIS SET: (16s,10p) -> [4s,3p]
    // Mg    S
    // 0.1172280000E+05       0.1977829317E-02
    // 0.1759930000E+04       0.1511399478E-01
    // 0.4008460000E+03       0.7391077448E-01
    // 0.1128070000E+03       0.2491909140E+00
    // 0.3599970000E+02       0.4879278316E+00
    // 0.1218280000E+02       0.3196618896E+00
    // Mg    SP
    // 0.1891800000E+03      -0.3237170471E-02       0.4928129921E-02
    // 0.4521190000E+02      -0.4100790597E-01       0.3498879944E-01
    // 0.1435630000E+02      -0.1126000164E+00       0.1407249977E+00
    // 0.5138860000E+01       0.1486330216E+00       0.3336419947E+00
    // 0.1906520000E+01       0.6164970898E+00       0.4449399929E+00
    // 0.7058870000E+00       0.3648290531E+00       0.2692539957E+00
    // Mg    SP
    // 0.9293400000E+00      -0.2122908985E+00      -0.2241918123E-01
    // 0.2690350000E+00      -0.1079854570E+00       0.1922708390E+00
    // 0.1173790000E+00       0.1175844977E+01       0.8461802916E+00
    // Mg    SP
    // 0.4210610000E-01       0.1000000000E+01       0.1000000000E+01
    // END

    /// Parses a string in NWChem format, returning a Basis631G.
    fn parse_nwchem(bstr: &str) -> Self {
        let mut basis = Basis631G {
            name: String::from(""),
            atomic_number: 0,
            basis_set: Vec::new()
        };

        // We'll track what element and shell we are currently filling
        let mut current_element: Option<String> = None;
        let mut current_shell_type: Option<ShellType> = None;
        let mut current_entries: ContractedGTO = ContractedGTO {
            primitives: Vec::new(),
            coefficients: Vec::new(),
            shell_type: String::from(""),
            n: 0,
            l: 0,
            m: 0,
            s: 0,
        };

        // Helper to push the "current shell" into `basis.data`
        // whenever we finish reading one.
        let mut push_current_shell = |elem: &str,
                                      shell_t: &ShellType,
                                      entries: &mut Vec<GTO>,
                                      basis: &mut Basis631G| {
            if !entries.is_empty() {
                let shell = ContractedGTO {
                    shell_type: shell_t.clone(),
                    entries: entries.drain(..).collect(),
                };
                basis.data.push((elem.to_string(), shell));
            }
        };

        // Go line by line
        for line in bstr.lines() {
            let line = line.trim();
            // Skip comments and blank lines
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            // If we hit "END", we’re done reading
            if line.eq_ignore_ascii_case("END") {
                // Push the last shell we were accumulating
                if let (Some(ref elem), Some(ref shell_t)) =
                    (current_element.as_ref(), current_shell_type.as_ref())
                {
                    push_current_shell(elem, shell_t, &mut current_entries, &mut basis);
                }
                break;
            }

            // Check if line looks like "Mg    S" or "Mg    SP" etc
            // Usually that means "Element  ShellType"
            // We'll assume first token is element, everything after is the shell type.
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() >= 2 {
                // Heuristic: If second token is S, P, D, SP, ...
                if let Some(shell_t) = parse_shell_type(tokens[1]) {
                    // That means we encountered a new shell definition line

                    // If we had an existing shell in progress, push it
                    if let (Some(ref elem), Some(ref shell_typ)) =
                        (current_element.as_ref(), current_shell_type.as_ref())
                    {
                        push_current_shell(elem, shell_typ, &mut current_entries, &mut basis);
                    }

                    // Start a new shell
                    current_element = Some(tokens[0].to_string());
                    current_shell_type = Some(shell_t);
                    // We’ll parse exponents/coeffs on subsequent lines
                    continue;
                }
            }

            // Otherwise, we assume it’s an exponent+coeff line.
            // E.g.  0.1172280000E+05       0.1977829317E-02
            // or    0.1891800000E+03      -0.3237170471E-02       0.4928129921E-02
            let numbers: Vec<f64> = line
                .split_whitespace()
                .filter_map(|s| parse_nwchem_float(s).ok())
                .collect();

            if !numbers.is_empty() {
                // By convention, the first number is exponent,
                // the rest are the coefficients for that row.
                let exponent = numbers[0];
                let coeffs = numbers[1..].to_vec();

                current_entries.push(ContractedGTO {
                    exponent,
                    coefficients: coeffs,
                });
            }
        }

        basis
    }
    fn new(bstr: String, format: BasisFormat) -> Self {
        match format {
            BasisFormat::NWChem => {
                let basis_set = Self::parse_nwchem(&bstr);
                Self {
                    name: "6-31G".to_string(),
                    atomic_number: 0,
                    basis_set,
                }
            }
            BasisFormat::Json => {
                let basis_set = Self::parse_json(&bstr);
                Self {
                    name: "6-31G".to_string(),
                    atomic_number: 0,
                    basis_set,
                }
            }
        }

        Basis631G {
            name: "6-31G".to_string(),
            atomic_number: 0,
            basis_set: Vec::new(),
        }
    }
}

impl Basis631G {
    // Serialize to pickle format
    pub fn to_pickle(&self) -> Result<Vec<u8>, serde_pickle::Error> {
        let options = serde_pickle::SerOptions::new();
        serde_pickle::to_vec(self, options)
    }

    // Deserialize from pickle format
    pub fn from_pickle(bytes: &[u8]) -> Result<Self, serde_pickle::Error> {
        let options = serde_pickle::DeOptions::new();
        serde_pickle::from_slice(bytes, options)
    }

    // Save to file in pickle format
    pub fn save_to_file(&self, filename: &str) -> std::io::Result<()> {
        let serialized = self.to_pickle()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        let mut file = File::create(filename)?;
        file.write_all(&serialized)
    }

    // Load from file in pickle format
    pub fn load_from_file(filename: &str) -> std::io::Result<Self> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        Self::from_pickle(&buffer)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))
    }
}
