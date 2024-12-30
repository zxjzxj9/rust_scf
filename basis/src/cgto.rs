/* Implement contracted gaussian type orbital (CGTO),
   based on gto.rs, which is the basic gaussian type orbital

   Author: Victor Zhang
   Date: 2024/12/31
*/

use std::fs::File;
use std::io::{Read, Write};
use nalgebra::Vector3;
use serde::{Deserialize, Serialize};
use serde_pickle;
// use serde_pickle::Serializer;
use crate::gto::GTO;
// use mendeleev::Element;
use periodic_table_on_an_enum;
use crate::basis::BasisFormat;


// need to consider how to reuse GTO integral, since s, p share the same exponents
#[derive(Debug, Serialize, Deserialize)]
pub struct ContractedGTO {
    pub primitives: Vec<GTO>,
    pub coefficients: Vec<f64>,
    // shell_type: 1s, 2s, 2px, 2py, 2pz, ...
    pub shell_type: String,
    pub n: i32, // 1, 2, ...
    pub l: i32, // 0 .. n-1
    pub m: i32, // -l .. +l
    pub s: i32, // +1 or -1, stand for alpha or beta, 0 stands for closed shell
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Basis631G {
    // define of the basis set
    pub name: String,
    // define atomic number
    pub atomic_number: u32,
    pub basis_set: Vec<ContractedGTO>,
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
    fn parse_primitive_block(lines: &[&str], center: Vector3<f64>, basis_type: &str) -> Vec<ContractedGTO> {
        let mut res: Vec<ContractedGTO> = Vec::new();

        match basis_type {
             "S" => {
                res.push(
                    ContractedGTO {
                        primitives: Vec::new(),
                        coefficients: Vec::new(),
                        shell_type: "1s".to_string(),
                        n: 1, l: 0, m: 0, s: 0,
                    }
                )
            }
             "SP" => {
                 let shells = vec![
                     ("1s", 0, 0, 0),
                     ("2px", 1, -1, 0),
                     ("2py", 1, 1, 0),
                     ("2pz", 1, 0, 0),
                 ];
                 for (shell_type, l, m, s) in shells {
                     res.push(ContractedGTO {
                         primitives: Vec::new(),
                         coefficients: Vec::new(),
                         shell_type: shell_type.to_string(),
                         n: 2, l, m, s,
                     });
                 }
            }
            _ => {
                panic!("Unsupported basis type: {}", basis_type);
            }
        }

        for line in lines {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() < 2 {
                continue;
            }

            // Parse the exponent and coefficients
            let alpha = tokens[0].parse::<f64>().unwrap();
            let s_coeff = tokens[1].parse::<f64>().unwrap();

            let s_gto = GTO::new(
                alpha,
                Vector3::new(0, 0, 0), // S orbital: l_xyz = [0,0,0]
                center,
            );
            res[0].primitives.push(s_gto);
            res[0].coefficients.push(s_coeff);


            // If this is an SP shell, also create P-type GTOs
            if basis_type == "SP" {
                // println!("test ### {:?}", tokens);
                let p_coeff = tokens[2].parse::<f64>().unwrap();
                let p_gto_x = GTO::new(alpha, Vector3::new(1, 0, 0), center);
                let p_gto_y = GTO::new(alpha, Vector3::new(0, 1, 0), center);
                let p_gto_z = GTO::new(alpha, Vector3::new(0, 0, 1), center);

                res[1].primitives.push(p_gto_x);
                res[1].coefficients.push(p_coeff);
                res[2].primitives.push(p_gto_y);
                res[2].coefficients.push(p_coeff);
                res[3].primitives.push(p_gto_z);
                res[3].coefficients.push(p_coeff);
            }
        }

        res
    }

    /// Parses a string in NWChem format, returning a Basis631G.
    pub fn parse_nwchem(input: &str) -> Self {
        let mut basis = Basis631G {
            name: String::from("6-31G"),
            atomic_number: 0,
            basis_set: Vec::new(),
        };

        let mut current_block = Vec::new();
        let mut current_shell_type = None;
        let center = Vector3::new(0.0, 0.0, 0.0); // Assuming center at origin

        for line in input.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let tokens: Vec<&str> = line.split_whitespace().collect();
            if tokens.len() >= 2 && (tokens[1] == "S" || tokens[1] == "SP") {
                // pase element name

                // Process previous block if it exists
                if !current_block.is_empty() {
                    // initialize the element information
                    let element = periodic_table_on_an_enum::Element::from_symbol(tokens[0]).unwrap();
                    basis.name = element.get_symbol().to_string();
                    basis.atomic_number = element.get_atomic_number() as u32;
                    let parsed = Self::parse_primitive_block(&current_block, center, current_shell_type.unwrap());
                    basis.basis_set.extend(parsed);
                    // Add to basis_set with appropriate shell type...
                    // You'll need to create ContractedGTO objects here
                }

                current_block.clear();
                current_shell_type = Some(tokens[1]);

            } else if !tokens.is_empty() && current_shell_type.is_some() {
                current_block.push(line);
            }
        }

        // Process the last block
        if !current_block.is_empty() {
            let parsed = Self::parse_primitive_block(&current_block, center, current_shell_type.unwrap());
            // Add to basis_set...
            basis.basis_set.extend(parsed);
        }

        basis
    }

    fn new(bstr: String, format: BasisFormat) -> Self {
        match format {
            BasisFormat::NWChem => {
                Self::parse_nwchem(&bstr);
            }
            BasisFormat::Json => {
                // not implemented
                todo!()
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
