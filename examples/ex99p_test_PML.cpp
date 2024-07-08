//                                MFEM Example 99 - Parallel Version
//
// Compile with: make ex99p
//
// Sample runs:  mpirun -np 4 ex99p -m ../data/inline-segment.mesh -o 3

//
// Device sample runs:
//               mpirun -np 4 ex99p -m ../data/inline-quad.mesh -o 3 -p 1 -pa -d cuda
//               mpirun -np 4 ex99p -m ../data/inline-hex.mesh -o 2 -p 2 -pa -d cuda
//               mpirun -np 4 ex99p -m ../data/star.mesh -r 1 -o 2 -sigma 10.0 -pa -d cuda
//
// Description:  This example code solves a simple electromagnetic wave
//               propagation problem corresponding to the second order
//               indefinite Maxwell equation
//
//                  (1/mu) * curl curl E - \omega^2 * epsilon E = f
//
//               with a Perfectly Matched Layer (PML).
//
//               The example demonstrates discretization with Nedelec finite
//               elements in 2D or 3D, as well as the use of complex-valued
//               bilinear and linear forms. Several test problems are included,
//               with prob = 0-3 having known exact solutions, see "On perfectly
//               matched layers for discontinuous Petrov-Galerkin methods" by
//               Vaziri Astaneh, Keith, Demkowicz, Comput Mech 63, 2019.
//
//               We recommend viewing Example 22 before viewing this example.

// added the anisotropic material from ex31
// added the point source
// added mesh from COMSOL
// add Laplace B.C. from ex27

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace mfem;

const auto MSH_FLT_PRECISION = std::numeric_limits<double>::max_digits10;

// Permittivity of free space [F/m].
constexpr double epsilon0_ = 8.8541878176e-12;

// Permeability of free space [H/m].
constexpr double mu0_ = 4.0e-7 * M_PI;

double mu_ = 1.0;
double epsilon_ = 1.0;
double sigma_ = 0.0;
double omega_ = 10.0;

bool check_for_inline_mesh(const char* mesh_file);

Mesh* LoadMeshNew(const std::string& path);

void source(const Vector& x, Vector& f);

void PrintMatrixConstantCoefficient(mfem::MatrixConstantCoefficient& coeff);

/// Convert a set of attribute numbers to a marker array
/** The marker array will be of size max_attr and it will contain only zeroes
    and ones. Ones indicate which attribute numbers are present in the attrs
    array. In the special case when attrs has a single entry equal to -1 the
    marker array will contain all ones. */
void AttrToMarker(ParMesh* pmesh, const Array<int>& attrs, Array<int>& marker);

void PrintArray2D(const Array2D<double>& arr);

// Class for setting up a simple Cartesian PML region
class PML
{
private:
   Mesh *mesh;

   int dim;

   // Length of the PML Region in each direction
   Array2D<double> length;

   // Computational Domain Boundary
   Array2D<double> comp_dom_bdr;

   // Domain Boundary
   Array2D<double> dom_bdr;

   // Integer Array identifying elements in the PML
   // 0: in the PML, 1: not in the PML
   Array<int> elems;

   // Compute Domain and Computational Domain Boundaries
   void SetBoundaries();

public:
   // Constructor
   PML(Mesh *mesh_,Array2D<double> length_);

   // Return Computational Domain Boundary
   Array2D<double> GetCompDomainBdr() {return comp_dom_bdr;}

   // Return Domain Boundary
   Array2D<double> GetDomainBdr() {return dom_bdr;}

   // Return Markers list for elements
   Array<int> * GetMarkedPMLElements() {return &elems;}

   // Mark elements in the PML region
   void SetAttributes(ParMesh *pmesh);
   void SetAttributes(ParMesh *pmesh, Array<int> pmlmaker_);

   // PML complex stretching function
   void StretchFunction(const Vector &x, vector<complex<double>> &dxs);
};

// Class for returning the PML coefficients of the bilinear form
class PMLDiagMatrixCoefficient : public VectorCoefficient
{
private:
   PML * pml = nullptr;
   void (*Function)(const Vector &, PML *, Vector &, int);
   int dim;
public:
   PMLDiagMatrixCoefficient(int dim_, void(*F)(const Vector &, PML *,
                                              Vector &, int),
                            PML * pml_)
      : VectorCoefficient(dim_), pml(pml_), Function(F), dim(dim_)
   {}

   using VectorCoefficient::Eval;

   virtual void Eval(Vector &K, ElementTransformation &T,
                     const IntegrationPoint &ip)
   {
      double x[3];
      Vector transip(x, 3);
      T.Transform(ip, transip);
      K.SetSize(vdim);
      (*Function)(transip, pml, K, dim);
   }
};

void maxwell_solution(const Vector &x, vector<complex<double>> &Eval);

void E_bdr_data_Re(const Vector &x, Vector &E);
void E_bdr_data_Im(const Vector &x, Vector &E);

void E_exact_Re(const Vector &x, Vector &E);
void E_exact_Im(const Vector &x, Vector &E);

void source(const Vector &x, Vector & f);

// Functions for computing the necessary coefficients after PML stretching.
// J is the Jacobian matrix of the stretching function
void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector & D, int dim);
void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector & D, int dim);
void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector & D, int dim);

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector & D, int dim);
void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector & D, int dim);
void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector & D, int dim);

Array2D<double> comp_domain_bdr;
Array2D<double> domain_bdr;

int dim;

int main(int argc, char* argv[])
{
//     {
//         int i=0;
//         while (0 == i){
// #ifdef _WIN32
//         Sleep(5000); // Windows API call, argument in milliseconds.
// #else
//         sleep(5); // Unix-like system call, argument in seconds.
// #endif
//         }
//     }
    // 0. Initialize MPI and HYPRE
    Mpi::Init(argc, argv);
    int num_procs = Mpi::WorldSize();
    int myid = Mpi::WorldRank();
    Hypre::Init();

    // 1. Parse command-line options.
    //const char* mesh_file = "../data/em_sphere_mfem_ex0_coarse.mphtxt";
    //const char* mesh_file = "../data/em_sphere_mfem_ex0.mphtxt";
    //const char* mesh_file = "../data/simple_cube.mphtxt";
    //const char* mesh_file = "../data/cube_comsol_pml.mphtxt";
     //const char* mesh_file = "../data/cube_comsol_coarse.mphtxt";
    const char* mesh_file = "../data/cube_comsol_ex_coarse.mphtxt";
    // const char* mesh_file = "../data/cube_comsol_rf1.mphtxt";
    //const char* mesh_file = "../data/inline-tet.mesh";
    int ser_ref_levels = 0;
    int par_ref_levels = 0;
    int order = 1;
    int prob = 1;
    double freq = 1200.0e6;
    double a_coef = 0.0;
    bool visualization = 1;
    bool herm_conv = true;
    bool slu_solver  = false;
    bool mumps_solver = false;
    bool exact_sol = true;
    bool pa = false;
    const char* device_config = "cpu";
    bool use_gmres = true;
    double mat_val = 1.0;
    double rbc_a_val = 1.0; // du/dn + a * u = b
    double rbc_b_val = 1.0;
    int logging_ = 1;
    bool comp_solver = true;
    int bprint = 1;

    std::vector<int> values = {1, 2, 3, 4, 5, 7, 8, 10, 11, 13, 14, \
        17, 20, 21, 23, 24, 27, 30, 31, 32, 33, 35, 36, 38, \
        41, 43, 46, 53, 56, 63, 64, 65, 66, 76, 77, 79, 82, \
        84, 87, 94, 97, 104, 105, 106, 107, 108, 109, 110, \
        111, 112, 113, 114, 115, 116};
    Array<int> abcs(values.data(), values.size());
    Array<int> dbcs;
    std::vector<int> values_pml = {1, 2, 3, 4, 5, 6, 7, 8, 9, \
        10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, \
        25, 26, 27, 28};
    Array<int> pmls(values_pml.data(), values_pml.size());

    OptionsParser args(argc, argv);
    args.AddOption(&mesh_file, "-m", "--mesh",
        "Mesh file to use.");
    args.AddOption(&ser_ref_levels, "-rs", "--refinements-serial",
                  "Number of serial refinements");
    args.AddOption(&par_ref_levels, "-rp", "--refinements-parallel",
                  "Number of parallel refinements");
    args.AddOption(&order, "-o", "--order",
        "Finite element order (polynomial degree).");
    args.AddOption(&prob, "-p", "--problem-type",
        "Choose between 0: H_1, 1: H(Curl), or 2: H(Div) "
        "damped harmonic oscillator.");
    args.AddOption(&a_coef, "-a", "--stiffness-coef",
        "Stiffness coefficient (spring constant or 1/mu).");
    args.AddOption(&epsilon_, "-b", "--mass-coef",
        "Mass coefficient (or epsilon).");
    args.AddOption(&sigma_, "-c", "--damping-coef",
        "Damping coefficient (or sigma).");
    args.AddOption(&mu_, "-mu", "--permeability",
        "Permeability of free space (or 1/(spring constant)).");
    args.AddOption(&epsilon_, "-eps", "--permittivity",
        "Permittivity of free space (or mass constant).");
    args.AddOption(&sigma_, "-sigma", "--conductivity",
        "Conductivity (or damping constant).");
    args.AddOption(&freq, "-f", "--frequency",
        "Frequency (in Hz).");
    args.AddOption(&herm_conv, "-herm", "--hermitian", "-no-herm",
        "--no-hermitian", "Use convention for Hermitian operators.");
#ifdef MFEM_USE_SUPERLU
    args.AddOption(&slu_solver, "-slu", "--superlu", "-no-slu",
                  "--no-superlu", "Use the SuperLU Solver.");
#endif
#ifdef MFEM_USE_MUMPS
    args.AddOption(&mumps_solver, "-mumps", "--mumps-solver", "-no-mumps",
                  "--no-mumps-solver", "Use the MUMPS Solver.");
#endif
    args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
        "--no-visualization",
        "Enable or disable GLVis visualization.");
    args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
        "--no-partial-assembly", "Enable Partial Assembly.");
    args.AddOption(&device_config, "-d", "--device",
        "Device configuration string, see Device::Configure().");
    args.AddOption(&use_gmres, "-g", "--gmres", "-no-g",
        "--no-grems", "Use GMRES solver or FGRMES solver.");
    args.AddOption(&mat_val, "-mat", "--material-value",
                  "Constant value for material coefficient "
                  "in the Laplace operator.");
    args.AddOption(&rbc_a_val, "-rbc-a", "--robin-a-value",
                  "Constant 'a' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
    args.AddOption(&rbc_b_val, "-rbc-b", "--robin-b-value",
                  "Constant 'b' value for Robin Boundary Condition: "
                  "du/dn + a * u = b.");
    args.AddOption(&abcs, "-abcs", "--absorbing-bc-surf",
        "Absorbing Boundary Condition Surfaces");
    args.AddOption(&pmls, "-pmls", "--pml-region",
        "Perfectly Matched Layer Regions");
    args.AddOption(&comp_solver, "-complex-sol", "--complex-solver",
        "-no-complex-sol", "--no-complex-solver", "Enable complex solver");
    args.Parse();
    if (slu_solver && mumps_solver)
    {
        if (myid == 0)
            cout << "WARNING: Both SuperLU and MUMPS have been selected,"
                << " please choose either one." << endl
                << "         Defaulting to SuperLU." << endl;
        mumps_solver = false;
    }

    if (!args.Good())
    {
        if (myid == 0)
        {
            args.PrintUsage(cout);
        }
        return 1;
    }
    if (myid == 0)
    {
        args.PrintOptions(cout);
    }

    MFEM_VERIFY(prob >= 0 && prob <= 2,
        "Unrecognized problem type: " << prob);

    if (a_coef != 0.0)
    {
        mu_ = 1.0 / a_coef;
    }
    if (freq > 0.0)
    {
        omega_ = 2.0 * M_PI * freq;
    }

    exact_sol = check_for_inline_mesh(mesh_file);
    if (myid == 0 && exact_sol)
    {
        cout << "Identified a mesh with known exact solution" << endl;
    }

    ComplexOperator::Convention conv =
        herm_conv ? ComplexOperator::HERMITIAN : ComplexOperator::BLOCK_SYMMETRIC;

    // 2. Enable hardware devices such as GPUs, and programming models such as
    //    CUDA, OCCA, RAJA and OpenMP based on command line options.
    Device device(device_config);
    if (myid == 0) { device.Print(); }

    // 3. Read the mesh from the given mesh file. We can handle triangular,
    //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes
    //    with the same code.
    //Mesh* mesh = new Mesh(mesh_file, 1, 1);
    Mesh* mesh = LoadMeshNew(mesh_file);
    dim = mesh->Dimension();

    Array2D<double> length(dim, 2);
    length = 0.1;

    PML* pml = new PML(mesh, length);
    comp_domain_bdr = pml->GetCompDomainBdr();
    domain_bdr = pml->GetDomainBdr();

    // 4. Refine the serial mesh to increase resolution. In this example we do
    //    'ref_levels' of uniform refinement where the user specifies
    //    the number of levels with the '-r' option.
    for (int l = 0; l < ser_ref_levels; l++)
    {
        mesh->UniformRefinement();
    }

    // 4a. Define a parallel mesh by a partitioning of the serial mesh.
    ParMesh *pmesh = new ParMesh(MPI_COMM_WORLD, *mesh);
    delete mesh;
    {
        for (int l = 0; l < par_ref_levels; l++)
        {
            pmesh->UniformRefinement();
        }
    }

    // 4b. Set element attributes in order to distinguish elements in the
    //    PML region
    // Setup PML length
    pml->SetAttributes(pmesh,pmls);

    // 5. Define a parallel finite element space on the parallel mesh. 
    //    Here we use continuous Lagrange, Nedelec, or Raviart-Thomas finite 
    //    elements of the specified order.
    if (dim == 1 && prob != 0)
    {
        if (myid == 0)
        {
            cout << "Switching to problem type 0, H1 basis functions, "
                << "for 1 dimensional mesh." << endl;
        }
        prob = 0;
    }

    FiniteElementCollection* fec = NULL;
    switch (prob)
    {
    case 0:  fec = new H1_FECollection(order, dim);      break;
    case 1:  fec = new ND_FECollection(order, dim);      break;
    case 2:  fec = new RT_FECollection(order - 1, dim);  break;
    default: break; // This should be unreachable
    }
    ParFiniteElementSpace* fespace = new ParFiniteElementSpace(pmesh, fec);
    HYPRE_BigInt size = fespace->GlobalTrueVSize();
    if (myid == 0)
    {
        cout << "Number of finite element unknowns: " << size << endl;
    }

    // 6. Determine the list of true (i.e. parallel conforming) essential boundary dofs.
    //    In this example, the boundary conditions are defined based on the type
    //    of mesh and the problem type.
    Array<int> ess_tdof_list;
    Array<int> ess_bdr;
    if (pmesh->bdr_attributes.Size())
    {
        ess_bdr.SetSize(pmesh->bdr_attributes.Max());
        AttrToMarker(pmesh, abcs, ess_bdr);
    }
    fespace->GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

    // 7. Set up the parallel linear form b(.) which corresponds to the right-hand side of
    //    the FEM linear system.
    VectorDeltaCoefficient* delta_one; // add point source
    double src_scalar = omega_ * 1.0;
    double position = 0.50;
    if (dim == 1)
    {
        Vector dir(1);
        dir[0] = 1.0;
        delta_one = new VectorDeltaCoefficient(dir, position, src_scalar);
    }
    else if (dim == 2)
    {
        Vector dir(2);
        dir[0] = 0.0; dir[1] = 1.0;
        delta_one = new VectorDeltaCoefficient(dir, position, position, src_scalar);
    }
    else if (dim == 3)
    {
        Vector dir(3);
        dir[0] = 0;
        dir[1] = 0;
        dir[2] = 1;
        delta_one = new VectorDeltaCoefficient(dir, position, position, position, src_scalar);
    }
    ParComplexLinearForm b(fespace, conv);
    VectorFunctionCoefficient f(dim, source);
    b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(*delta_one)); // add delta point source
    // b.AddDomainIntegrator(NULL, new VectorFEDomainLFIntegrator(f)); // add Gaussian point source
    //b.AddBoundaryIntegrator(new VectorBoundaryLFIntegrator(m_rbcBCoef), NULL, rbc_bdr);
    b.Vector::operator=(0.0);
    // Access and print the imaginary part
    
    //imagPart.Print();

    b.Assemble();

    // Access and print the real part
    //const Vector& imagPart = b.imag();
    //imagPart.Print();

    // 8. Define the solution vector u as a parallel complex finite element grid function
    //    corresponding to fespace. Initialize u with initial guess of 1+0i or
    //    the exact solution if it is known.
    ParComplexGridFunction u(fespace);
    u = 0.0;

    ConstantCoefficient zeroCoef(0.0);
    ConstantCoefficient oneCoef(1.0);

    Vector zeroVec(dim); zeroVec = 0.0;
    Vector oneVec(dim);  oneVec = 0.0; oneVec[(prob == 2) ? (dim - 1) : 0] = 1.0;
    VectorConstantCoefficient zeroVecCoef(zeroVec);
    VectorConstantCoefficient oneVecCoef(oneVec);

    // 9. Set up the sesquilinear form a(.,.) on the finite element space
    //    corresponding to the damped harmonic oscillator operator of the
    //    appropriate type:
    //
    //    0) A scalar H1 field
    //       -Div(a Grad) - omega^2 b + i omega c
    //
    //    1) A vector H(Curl) field
    //       Curl(a Curl) - omega^2 b + i omega c
    //
    //    2) A vector H(Div) field
    //       -Grad(a Div) - omega^2 b + i omega c
    //
    ConstantCoefficient stiffnessCoef(1.0 / mu_ / mu0_);
    ConstantCoefficient massCoef(-omega_ * omega_ * epsilon_ * epsilon0_); std::cout << std::setw(20) << ( - omega_ * omega_ * epsilon_ * epsilon0_) << std::endl;
    ConstantCoefficient lossCoef(omega_ * sigma_); std::cout << std::setw(20) << (omega_ * sigma_) << std::endl;
    ConstantCoefficient negMassCoef(omega_ * omega_ * epsilon_ * epsilon0_);

    DenseMatrix sigmaMat(3);
    sigmaMat(0, 0) = 3.0; sigmaMat(1, 1) = 2.0; sigmaMat(2, 2) = 4.0;
    sigmaMat(0, 2) = 0.0; sigmaMat(2, 0) = 0.0;
    sigmaMat(0, 1) = M_SQRT1_2; sigmaMat(1, 0) = M_SQRT1_2; // 1/sqrt(2) in cmath
    sigmaMat(1, 2) = M_SQRT1_2; sigmaMat(2, 1) = M_SQRT1_2;
    Vector omega(dim); omega = omega_;
    sigmaMat.LeftScaling(omega);
    MatrixConstantCoefficient aniLossCoef(sigmaMat);
    //PrintMatrixConstantCoefficient(aniLossCoef);

    DenseMatrix epsilonMat(3);
    epsilonMat(0, 0) = 2.0; epsilonMat(1, 1) = 3.0; epsilonMat(2, 2) = 4.0;
    epsilonMat(0, 2) = 0.0; epsilonMat(2, 0) = 0.0;
    epsilonMat(0, 1) = 0.0; epsilonMat(1, 0) = 0.0; // 1/sqrt(2) in cmath
    epsilonMat(1, 2) = 0.0; epsilonMat(2, 1) = 0.0;
    omega = -omega_ * omega_ * epsilon0_;
    epsilonMat.LeftScaling(omega);
    MatrixConstantCoefficient aniMassCoef(epsilonMat);
    // PrintMatrixConstantCoefficient(aniMassCoef);

    Array<int> attr; // active attr for domain
    Array<int> attrPML; // active attr for PML
    if (pmesh->attributes.Size())
    {
        attr.SetSize(pmesh->attributes.Max());
        attrPML.SetSize(pmesh->attributes.Max());
        attr = 1;
        attrPML = 0;
        if (pmesh->attributes.Max() > 1)
            for(int i = 0; i<pmesh->attributes.Max(); ++i)
            {
                bool found = false;
                for (int j = 0; j < pmls.Size(); j++) {
                    if (i == pmls[j]-1) {
                        found = true;
                        break;
                    }
                }
                if (found) {
                    attr[i] = 0;
                    attrPML[i] = 1;
                }
            }
    }

    if (myid == 0)
    {
        for (int i = 0; i < attr.Size(); i++)
        {
            std::cout << i << "th domain: " << attr[i] << " " << std::endl;
            std::cout << i << "th PML: " << attrPML[i] << " " << std::endl;
        }
        std::cout << std::endl;

        // Print comp_domain_bdr
        std::cout << "comp_domain_bdr:" << std::endl;
        PrintArray2D(comp_domain_bdr);

        // Print domain_bdr
        std::cout << "domain_bdr:" << std::endl;
        PrintArray2D(domain_bdr);
    }   

    ConstantCoefficient muinv(1.0 / mu_ / mu0_);
    ConstantCoefficient omeg(-pow(omega_, 2) * epsilon_ * epsilon0_);
    ConstantCoefficient loss(omega_ * sigma_);
    RestrictedCoefficient restr_muinv(muinv,attr);
    RestrictedCoefficient restr_omeg(omeg,attr);
    RestrictedCoefficient restr_loss(loss,attr);

    int cdim = (dim == 2) ? 1 : dim;
    PMLDiagMatrixCoefficient pml_c1_Re(cdim,detJ_inv_JT_J_Re, pml);
    PMLDiagMatrixCoefficient pml_c1_Im(cdim,detJ_inv_JT_J_Im, pml);
    ScalarVectorProductCoefficient c1_Re(muinv,pml_c1_Re);
    ScalarVectorProductCoefficient c1_Im(muinv,pml_c1_Im);
    VectorRestrictedCoefficient restr_c1_Re(c1_Re,attrPML);
    VectorRestrictedCoefficient restr_c1_Im(c1_Im,attrPML);

    PMLDiagMatrixCoefficient pml_c2_Re(dim, detJ_JT_J_inv_Re,pml);
    PMLDiagMatrixCoefficient pml_c2_Im(dim, detJ_JT_J_inv_Im,pml);
    ScalarVectorProductCoefficient c2_Re(omeg,pml_c2_Re);
    ScalarVectorProductCoefficient c2_Im(omeg,pml_c2_Im);
    VectorRestrictedCoefficient restr_c2_Re(c2_Re,attrPML);
    VectorRestrictedCoefficient restr_c2_Im(c2_Im,attrPML);

    ParSesquilinearForm* a = new ParSesquilinearForm(fespace, conv);
    if (pa) { a->SetAssemblyLevel(AssemblyLevel::PARTIAL); }
    switch (prob)
    {
    case 0:
        a->AddDomainIntegrator(new DiffusionIntegrator(stiffnessCoef),
            NULL);
        a->AddDomainIntegrator(new MassIntegrator(massCoef),
            new MassIntegrator(lossCoef));
        break;
    case 1:
        a->AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv),
            NULL);
        a->AddDomainIntegrator(new VectorFEMassIntegrator(restr_omeg),
            NULL);
        // Integrators inside the PML region
        a->AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_Re),
                                new CurlCurlIntegrator(restr_c1_Im));
        a->AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_Re),
                                new VectorFEMassIntegrator(restr_c2_Im));
        break;
    case 2:
        a->AddDomainIntegrator(new DivDivIntegrator(stiffnessCoef),
            NULL);
        a->AddDomainIntegrator(new VectorFEMassIntegrator(massCoef),
            new VectorFEMassIntegrator(lossCoef));
        break;
    default: break; // This should be unreachable
    }

    // 10. Assemble the parallel bilinear form and the corresponding linear
    //     system, applying any necessary transformations such as: parallel 
    //     assembly, eliminating boundary conditions, conforming constraints
    //     for non-conforming AMR, etc.
    //a->SetDiagonalPolicy(mfem::Operator::DiagonalPolicy::DIAG_KEEP);
    a->Assemble();

    OperatorPtr Ah;
    Vector B, U;

    a->FormLinearSystem(ess_tdof_list, u, b, Ah, U, B);
    if (myid == 0)
    {
        std::cout << "Size of linear system: " << Ah->Width() << endl << endl;
        //std::cout << "Printing Matrix A..." << std::endl;
        //std::ofstream A_file("Asp_matrix.txt");
        //A->PrintMatlab(A_file);
    }

    //ComplexSparseMatrix* Asp_blk = a->AssembleComplexSparseMatrix();
    //SparseMatrix* Asp = Asp_blk->GetSystemMatrix();

    // 11. Solve using a direct or an iterative solver
#ifdef MFEM_USE_SUPERLU
    if (!pa && slu_solver)
    {
        // Transform to monolithic HypreParMatrix
        HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
        SuperLURowLocMatrix SA(*A);
        SuperLUSolver superlu(MPI_COMM_WORLD);
        superlu.SetPrintStatistics(false);
        superlu.SetSymmetricPattern(false);
        superlu.SetColumnPermutation(superlu::PARMETIS);
        superlu.SetOperator(SA);
        superlu.Mult(B, X);
        delete A;
    }
#endif
#ifdef MFEM_USE_MUMPS
    if (!pa && mumps_solver)
    {
        HypreParMatrix *A = Ah.As<ComplexHypreParMatrix>()->GetSystemMatrix();
        MUMPSSolver mumps(A->GetComm());
        mumps.SetPrintLevel(0);
        mumps.SetMatrixSymType(MUMPSSolver::MatType::UNSYMMETRIC);
        mumps.SetOperator(*A);
        mumps.Mult(B, X);
        delete A;
    }
#endif
    // 11a. Set up the parallel Bilinear form a(.,.) for the preconditioner
    //
    //    In Comp
    //    Domain:   1/mu (Curl E, Curl F) + omega^2 * epsilon (E,F)
    //
    //    In PML:   1/mu (abs(1/det(J) J^T J) Curl E, Curl F)
    //              + omega^2 * epsilon (abs(det(J) * (J^T J)^-1) * E, F)
    if (pa || (!slu_solver && !mumps_solver))
    {
      ConstantCoefficient absomeg(pow(omega_, 2) * epsilon_);
      RestrictedCoefficient restr_absomeg(absomeg,attr);

      ParBilinearForm prec(fespace);
      prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_muinv));
      prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_absomeg));

      PMLDiagMatrixCoefficient pml_c1_abs(cdim,detJ_inv_JT_J_abs, pml);
      ScalarVectorProductCoefficient c1_abs(muinv,pml_c1_abs);
      VectorRestrictedCoefficient restr_c1_abs(c1_abs,attrPML);

      PMLDiagMatrixCoefficient pml_c2_abs(dim, detJ_JT_J_inv_abs,pml);
      ScalarVectorProductCoefficient c2_abs(absomeg,pml_c2_abs);
      VectorRestrictedCoefficient restr_c2_abs(c2_abs,attrPML);

      prec.AddDomainIntegrator(new CurlCurlIntegrator(restr_c1_abs));
      prec.AddDomainIntegrator(new VectorFEMassIntegrator(restr_c2_abs));

      if (pa) { prec.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
      prec.Assemble();

      // 16b. Define and apply a parallel GMRES solver for AU=B with a block
      //      diagonal preconditioner based on hypre's AMS preconditioner.
      Array<int> offsets(3);
      offsets[0] = 0;
      offsets[1] = fespace->GetTrueVSize();
      offsets[2] = fespace->GetTrueVSize();
      offsets.PartialSum();

      std::unique_ptr<Operator> pc_r;
      std::unique_ptr<Operator> pc_i;
      int s = (conv == ComplexOperator::HERMITIAN) ? -1.0 : 1.0;
      if (pa)
      {
         // Jacobi Smoother
         pc_r.reset(new OperatorJacobiSmoother(prec, ess_tdof_list));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }
      else
      {
         OperatorPtr PCOpAh;
         prec.FormSystemMatrix(ess_tdof_list, PCOpAh);

         // Hypre AMS
         pc_r.reset(new HypreAMS(*PCOpAh.As<HypreParMatrix>(), fespace));
         pc_i.reset(new ScaledOperator(pc_r.get(), s));
      }

      BlockDiagonalPreconditioner BlockDP(offsets);
      BlockDP.SetDiagonalBlock(0, pc_r.get());
      BlockDP.SetDiagonalBlock(1, pc_i.get());

      GMRESSolver gmres(MPI_COMM_WORLD);
      gmres.SetPrintLevel(1);
      gmres.SetKDim(200);
      gmres.SetMaxIter(pa ? 5000 : 2000);
      gmres.SetRelTol(1e-4);
      gmres.SetAbsTol(0.0);
      gmres.SetOperator(*Ah);
      gmres.SetPreconditioner(BlockDP);
      gmres.Mult(B, U);
   }
   

    // 12. Recover the parallel solution as a finite element grid function and compute the
    //     errors if the exact solution is known.
    a->RecoverFEMSolution(U, b, u);

    //std::cout << "Printing solution u..." << std::endl;
    //std::ofstream u_file("u_field.txt");
    //u.Print(u_file);
    // 13. Save the refined mesh and the solution in parallel. This output can be viewed
    //     later using GLVis: "glvis -np <np> -m mesh -g sol".
    {
      ostringstream mesh_name, sol_r_name, sol_i_name;
      mesh_name << "mesh." << setfill('0') << setw(6) << myid;
      sol_r_name << "ex99p-sol_r." << setfill('0') << setw(6) << myid;
      sol_i_name << "ex99p-sol_i." << setfill('0') << setw(6) << myid;

      ofstream mesh_ofs(mesh_name.str().c_str());
      mesh_ofs.precision(8);
      pmesh->Print(mesh_ofs);

      ofstream sol_r_ofs(sol_r_name.str().c_str());
      ofstream sol_i_ofs(sol_i_name.str().c_str());
      sol_r_ofs.precision(8);
      sol_i_ofs.precision(8);
      u.real().Save(sol_r_ofs);
      u.imag().Save(sol_i_ofs);
   }

    // 14. Send the solution by socket to a GLVis server.
    if (visualization)
   {
      // Define visualization keys for GLVis (see GLVis documentation)
      string keys;
      keys = (dim == 3) ? "keys macF\n" : keys = "keys amrRljcUUuu\n";

      char vishost[] = "localhost";
      int visport = 19916;

      {
         socketstream sol_sock_re(vishost, visport);
         sol_sock_re.precision(8);
         sol_sock_re << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << u.real() << keys
                     << "window_title 'Solution real part'" << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }

      {
         socketstream sol_sock_im(vishost, visport);
         sol_sock_im.precision(8);
         sol_sock_im << "parallel " << num_procs << " " << myid << "\n"
                     << "solution\n" << *pmesh << u.imag() << keys
                     << "window_title 'Solution imag part'" << flush;
         MPI_Barrier(MPI_COMM_WORLD); // try to prevent streams from mixing
      }

      {
         ParGridFunction u_t(fespace);
         u_t = u.real();

         socketstream sol_sock(vishost, visport);
         sol_sock.precision(8);
         sol_sock << "parallel " << num_procs << " " << myid << "\n"
                  << "solution\n" << *pmesh << u_t << keys << "autoscale off\n"
                  << "window_title 'Harmonic Solution (t = 0.0 T)'"
                  << "pause\n" << flush;

         if (myid == 0)
         {
            cout << "GLVis visualization paused."
                 << " Press space (in the GLVis window) to resume it.\n";
         }

         int num_frames = 32;
         int i = 0;
         while (sol_sock)
         {
            double t = (double)(i % num_frames) / num_frames;
            ostringstream oss;
            oss << "Harmonic Solution (t = " << t << " T)";

            add(cos(2.0*M_PI*t), u.real(), sin(2.0*M_PI*t), u.imag(), u_t);
            sol_sock << "parallel " << num_procs << " " << myid << "\n";
            sol_sock << "solution\n" << *pmesh << u_t
                     << "window_title '" << oss.str() << "'" << flush;
            i++;
         }
      }
   }

    // 14b. Save data in the ParaView format
    ParaViewDataCollection paraview_dc("ex99p_PML", pmesh);
    paraview_dc.SetDataFormat(VTKFormat::ASCII);
    paraview_dc.SetPrefixPath("ParaView");
    paraview_dc.SetLevelsOfDetail(order>1 ? order-1 : order);
    //paraview_dc.SetCycle(0);
    paraview_dc.SetDataFormat(VTKFormat::BINARY);
    //paraview_dc.SetHighOrderOutput(true);
    paraview_dc.SetTime(0.0); // set the time
    paraview_dc.RegisterField("real", &u.real());
    paraview_dc.RegisterField("imag", &u.imag());
    paraview_dc.Save();

    // 15. Free the used memory.
    delete pml;
    delete fespace;
    delete fec;
    delete pmesh;
    return 0;
}

void source(const Vector& x, Vector& f)
{
    Vector center(dim);
    double r = 0.0;
    for (int i = 0; i < dim; ++i)
    {
        center(i) = 0.5 * (comp_domain_bdr(i, 0) + comp_domain_bdr(i, 1));
        r += pow(x[i] - center[i], 2.);
    }
    double n = 500.0 * omega_ * sqrt(epsilon0_ * epsilon_ * mu0_ * mu_) / M_PI;
    double coeff = pow(n, 2) / M_PI;
    double alpha = -pow(n, 2) * r;
    f = 0.0;
    f[1] = coeff * exp(alpha);
}

bool check_for_inline_mesh(const char* mesh_file)
{
    string file(mesh_file);
    size_t p0 = file.find_last_of("/");
    string s0 = file.substr((p0 == string::npos) ? 0 : (p0 + 1), 7);
    return s0 == "inline-";
}

Mesh* LoadMeshNew(const std::string& path)
{
    // Read the (serial) mesh from the given mesh file. Handle preparation for refinement and
    // orientations here to avoid possible reorientations and reordering later on. MFEM
    // supports a native mesh format (.mesh), VTK/VTU, Gmsh, as well as some others. We use
    // built-in converters for the types we know, otherwise rely on MFEM to do the conversion
    // or error out if not supported.
    std::filesystem::path mfile(path);
    if (mfile.extension() == ".mphtxt" || mfile.extension() == ".mphbin" ||
        mfile.extension() == ".nas" || mfile.extension() == ".bdf")
    {
        // Put translated mesh in temporary string buffer.
        std::stringstream fi(std::stringstream::in | std::stringstream::out);
        // fi << std::fixed;
        fi << std::scientific;
        fi.precision(MSH_FLT_PRECISION);

        Mesh* tempmesh = new Mesh();

        if (mfile.extension() == ".mphtxt" || mfile.extension() == ".mphbin")
        {
            tempmesh->ConvertMeshComsol(path, fi);
        }
        else
        {
            tempmesh->ConvertMeshNastran(path, fi);
        }

        return new Mesh(fi, 1, 1, true);
    }
    // Otherwise, just rely on MFEM load the mesh.
    named_ifgzstream fi(path);
    if (!fi.good())
    {
        MFEM_ABORT("Unable to open mesh file \"" << path << "\"!");
    }
    Mesh* mesh = new Mesh(fi, 1, 1, true);
    mesh->EnsureNodes();
    return mesh;
}

void PrintMatrixConstantCoefficient(mfem::MatrixConstantCoefficient& coeff)
{
    const mfem::DenseMatrix& mat = coeff.GetMatrix();
    int height = mat.Height();
    int width = mat.Width();

    // Set the output format
    std::cout << std::scientific << std::setprecision(4) << std::right;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            std::cout << std::setw(16) << mat(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to print Array2D<double>
void PrintArray2D(const mfem::Array2D<double>& arr)
{
    for (int i = 0; i < arr.NumRows(); i++)
    {
        for (int j = 0; j < arr.NumCols(); j++)
        {
            std::cout << arr(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void AttrToMarker(ParMesh *pmesh, const Array<int>& attrs, Array<int>& marker)
{
    if (attrs.Size() == 1 && attrs[0] == -1)
    {
        marker = 1;
    }
    else
    {
        marker = 0;
        for (int j = 0; j < pmesh->GetNBE(); j++)
        {
            int k = pmesh->GetBdrAttribute(j);
            // std::cout << k << std::endl;
            for (int l = 0; l < attrs.Size(); l++) {
                if (k == attrs[l]) {
                    marker[k - 1] = 1;
                    //std::cout << "pml is true." << std::endl;
                    break;
                }
            }
        }
    }
}


void detJ_JT_J_inv_Re(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).real();
   }
}

void detJ_JT_J_inv_Im(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = (det / pow(dxs[i], 2)).imag();
   }
}

void detJ_JT_J_inv_abs(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   for (int i = 0; i < dim; ++i)
   {
      D(i) = abs(det / pow(dxs[i], 2));
   }
}

void detJ_inv_JT_J_Re(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det(1.0, 0.0);
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   // in the 2D case the coefficient is scalar 1/det(J)
   if (dim == 2)
   {
      D = (1.0 / det).real();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).real();
      }
   }
}

void detJ_inv_JT_J_Im(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = (1.0 / det).imag();
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = (pow(dxs[i], 2) / det).imag();
      }
   }
}

void detJ_inv_JT_J_abs(const Vector &x, PML * pml, Vector &D, int dim)
{
   vector<complex<double>> dxs(dim);
   complex<double> det = 1.0;
   pml->StretchFunction(x, dxs);

   for (int i = 0; i < dim; ++i)
   {
      det *= dxs[i];
   }

   if (dim == 2)
   {
      D = abs(1.0 / det);
   }
   else
   {
      for (int i = 0; i < dim; ++i)
      {
         D(i) = abs(pow(dxs[i], 2) / det);
      }
   }
}

PML::PML(Mesh *mesh_, Array2D<double> length_)
   : mesh(mesh_), length(length_)
{
   dim = mesh->Dimension();
   SetBoundaries();
}

void PML::SetBoundaries()
{
   comp_dom_bdr.SetSize(dim, 2);
   dom_bdr.SetSize(dim, 2);
   Vector pmin, pmax;
   mesh->GetBoundingBox(pmin, pmax);
   for (int i = 0; i < dim; i++)
   {
      dom_bdr(i, 0) = pmin(i);
      dom_bdr(i, 1) = pmax(i);
      comp_dom_bdr(i, 0) = dom_bdr(i, 0) + length(i, 0);
      comp_dom_bdr(i, 1) = dom_bdr(i, 1) - length(i, 1);
   }
}

void PML::SetAttributes(ParMesh *pmesh)
{
   // Initialize bdr attributes
   for (int i = 0; i < pmesh->GetNBE(); ++i)
   {
      pmesh->GetBdrElement(i)->SetAttribute(i+1);
   }

   int nrelem = pmesh->GetNE();

   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
      elems[i] = 1;
      bool in_pml = false;
      Element *el = pmesh->GetElement(i);
      Array<int> vertices;

      // Initialize attribute
      el->SetAttribute(1);
      el->GetVertices(vertices);
      int nrvert = vertices.Size();

      // Check if any vertex is in the PML
      for (int iv = 0; iv < nrvert; ++iv)
      {
         int vert_idx = vertices[iv];
         double *coords = pmesh->GetVertex(vert_idx);
         for (int comp = 0; comp < dim; ++comp)
         {
            if (coords[comp] > comp_dom_bdr(comp, 1) ||
                coords[comp] < comp_dom_bdr(comp, 0))
            {
               in_pml = true;
               break;
            }
         }
      }
      if (in_pml)
      {
         elems[i] = 0;
         el->SetAttribute(2);
      }
   }
   pmesh->SetAttributes();
}

void PML::SetAttributes(ParMesh *pmesh, Array<int> pmlmaker_)
{
   int nrelem = pmesh->GetNE();

   // Initialize list with 1
   elems.SetSize(nrelem);

   // Loop through the elements and identify which of them are in the PML
   for (int i = 0; i < nrelem; ++i)
   {
        elems[i] = 1;
        bool in_pml = false;
        Element *el = pmesh->GetElement(i);

        // Check if element attribute is in the PML
        int j = el->GetAttribute();
        //std::cout << "element: " << i << ", attribute : " << j << std::endl;
        for (int k = 0; k < pmlmaker_.Size(); k++) {
            if (j == pmlmaker_[k]) {
                in_pml = true;
                //std::cout << "pml is true." << std::endl;
                break;
            }
        }
        if (in_pml)
        {
            elems[i] = 0;
            el->SetAttribute(2);
        }
        else
        {
            el->SetAttribute(1);
        }
   }
   pmesh->SetAttributes();
}

void PML::StretchFunction(const Vector &x,
                          vector<complex<double>> &dxs)
{
   complex<double> zi = complex<double>(0., 1.);

   double n = 2.0;
   double c = 5.0;
   double coeff;
   double k = omega_ * sqrt(epsilon0_ * epsilon_ * mu_ * mu0_);

   // Stretch in each direction independently
   for (int i = 0; i < dim; ++i)
   {
      dxs[i] = 1.0;
      if (x(i) >= comp_domain_bdr(i, 1))
      {
         coeff = n * c / k / pow(length(i, 1), n);
         dxs[i] = 1.0 + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 1), n - 1.0));
      }
      if (x(i) <= comp_domain_bdr(i, 0))
      {
         coeff = n * c / k / pow(length(i, 0), n);
         dxs[i] = 1.0 + zi * coeff *
                  abs(pow(x(i) - comp_domain_bdr(i, 0), n - 1.0));
      }
   }
}
