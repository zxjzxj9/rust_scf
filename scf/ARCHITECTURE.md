# SCF Code Architecture

## Module Dependency Graph

```
┌─────────────────────────────────────────────────────────────┐
│                         main.rs                              │
│                   (CLI Entry Point)                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ├─────────────────────────────────────────────────┐
             │                                                 │
             v                                                 v
┌────────────────────────┐                      ┌─────────────────────────┐
│   config/              │                      │   io/                   │
│   ├── mod.rs           │                      │   ├── mod.rs            │
│   └── args.rs          │                      │   ├── output.rs         │
│   (Configuration)      │                      │   └── basis_loader.rs   │
└────────────────────────┘                      └─────────────────────────┘
             │                                                 │
             │                                                 │
             v                                                 v
┌──────────────────────────────────────────────────────────────────────────┐
│                            scf_impl/                                      │
│   ├── mod.rs          (SCF trait, DIIS)                                  │
│   ├── simple.rs       (Restricted Hartree-Fock)                          │
│   ├── simple_spin.rs  (Unrestricted Hartree-Fock)                        │
│   └── tests.rs        (Unit tests)                                       │
└──────────────────────────────────────────────────────────────────────────┘
             │
             │
             v
┌──────────────────────────────────────────────────────────────────────────┐
│                           optim_impl/                                     │
│   ├── mod.rs              (GeometryOptimizer trait)                      │
│   ├── cg.rs               (Conjugate Gradient)                           │
│   └── steepest_descent.rs (Steepest Descent)                             │
└──────────────────────────────────────────────────────────────────────────┘
             │
             │
             v
┌──────────────────────────────────────────────────────────────────────────┐
│                      force_validation.rs                                  │
│                   (Force Calculation Validation)                          │
└──────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### SCF Calculation Flow

```
User Input (YAML)
      │
      v
   config/args.rs ──> Parse CLI arguments
      │
      v
   config/mod.rs ──> Load & validate configuration
      │
      v
   io/basis_loader.rs ──> Fetch basis sets
      │
      v
   main.rs ──> Prepare geometry & basis
      │
      ├──> SimpleSCF (RHF)
      │    └──> scf_impl/simple.rs
      │         └──> SCF cycle
      │
      └──> SpinSCF (UHF)
           └──> scf_impl/simple_spin.rs
                └──> SCF cycle
      │
      v
   Converged Energy & Orbitals
      │
      v (optional)
   optim_impl/ ──> Geometry Optimization
      │
      v
   io/output.rs ──> Format & write results
```

### Optimization Flow

```
Initial Geometry
      │
      v
   GeometryOptimizer::init()
      │
      v
   SCF Calculation ──> Energy & Forces
      │
      v
   Update Coordinates
      │   │
      │   └──> CG: calculate beta, update direction
      │   └──> SD: step in force direction
      │
      v
   Check Convergence?
      │
      ├──> No: repeat cycle
      │
      └──> Yes: Return optimized geometry
```

## Module Responsibilities

### config/
**Purpose**: Configuration management
- Parse command-line arguments
- Load and validate YAML configuration
- Provide default values
- Merge CLI and file configurations

**Dependencies**: `clap`, `serde`, `serde_yml`

### io/
**Purpose**: Input/output operations
- Load basis sets from online database
- Setup logging infrastructure
- Format output results
- Handle file I/O

**Dependencies**: `reqwest`, `tracing`, `tracing-subscriber`

### scf_impl/
**Purpose**: SCF calculation algorithms
- Define SCF trait interface
- Implement Restricted Hartree-Fock (RHF)
- Implement Unrestricted Hartree-Fock (UHF)
- DIIS convergence acceleration
- Force calculations

**Dependencies**: `nalgebra`, `basis`, `rayon`, `periodic_table_on_an_enum`

### optim_impl/
**Purpose**: Geometry optimization algorithms
- Define GeometryOptimizer trait
- Implement Conjugate Gradient method
- Implement Steepest Descent method
- Line search and convergence checking
- Force validation

**Dependencies**: `nalgebra`, `periodic_table_on_an_enum`, `tracing`

### force_validation.rs
**Purpose**: Validation utilities
- Numerical force calculation
- Force comparison and validation
- Optimization testing utilities
- Diagnostic tools

**Dependencies**: `nalgebra`, `basis`, `periodic_table_on_an_enum`, `tracing`

## Design Principles Applied

### 1. **Separation of Concerns**
Each module has a single, well-defined responsibility.

### 2. **Dependency Inversion**
High-level modules (main.rs) depend on abstractions (traits), not concrete implementations.

### 3. **Single Responsibility Principle**
Each file/module does one thing and does it well.

### 4. **DRY (Don't Repeat Yourself)**
Common functionality extracted into reusable modules.

### 5. **Modularity**
Modules can be developed, tested, and modified independently.

## Testing Strategy

```
Unit Tests
├── scf_impl/tests.rs
│   ├── SimpleSCF tests
│   ├── SpinSCF tests
│   └── Force calculation tests
│
└── optim_impl/
    ├── CG optimizer tests (in cg.rs)
    └── SD optimizer tests (in steepest_descent.rs)

Integration Tests
└── force_validation.rs
    ├── Numerical vs analytical forces
    ├── Optimization convergence
    └── Algorithm comparison
```

## Extension Points

### Adding a New SCF Method
1. Create new file in `scf_impl/` (e.g., `mp2.rs`)
2. Implement the `SCF` trait
3. Export from `scf_impl/mod.rs`
4. Update main.rs routing logic

### Adding a New Optimizer
1. Create new file in `optim_impl/` (e.g., `bfgs.rs`)
2. Implement the `GeometryOptimizer` trait
3. Add to factory in `optim_impl/mod.rs`
4. Update CLI arguments in `config/args.rs`

### Adding a New I/O Format
1. Add reader/writer in `io/` module
2. Update configuration parsing if needed
3. Export from `io/mod.rs`

## Performance Considerations

- **Parallelization**: `rayon` used in SCF matrix operations
- **Memory**: Sparse matrix operations where possible
- **Caching**: Basis sets cached after first fetch
- **Optimization**: Profile-guided optimization possible due to modular structure

## Future Architecture Enhancements

1. **Plugin System**: Allow external optimizers/SCF methods
2. **State Management**: Centralized state for complex workflows
3. **Async I/O**: Non-blocking basis set fetching
4. **Result Caching**: Cache SCF results for repeated geometries
5. **Workspace Pattern**: Split into multiple crates for very large projects

