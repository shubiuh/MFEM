//                                MFEM Example 0
//
// Compile with: make ex0
//
// Sample runs:  ex0
//               ex0 -m ../data/fichera.mesh
//               ex0 -m ../data/square-disc.mesh -o 2
//
// Description: This example code demonstrates the most basic usage of MFEM to
//              define a simple finite element discretization of the Laplace
//              problem -Delta u = 1 with zero Dirichlet boundary conditions.
//              General 2D/3D mesh files and finite element polynomial degrees
//              can be specified by command line options.

#include "mfem.hpp"
#include "meshio.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <array>
#include <limits>
#include <map>
#include <filesystem>

using namespace std;
using namespace mfem;

const auto MSH_FLT_PRECISION = std::numeric_limits<double>::max_digits10;

Mesh LoadMeshNew(const std::string &path)
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

		if (mfile.extension() == ".mphtxt" || mfile.extension() == ".mphbin")
		{
			palace::mesh::ConvertMeshComsol(path, fi);
			// mesh::ConvertMeshComsol(path, fo);
		}
		else
		{
			palace::mesh::ConvertMeshNastran(path, fi);
			// mesh::ConvertMeshNastran(path, fo);
		}

		return Mesh(fi, 1, 1, true);
	}
	// Otherwise, just rely on MFEM load the mesh.
	named_ifgzstream fi(path);
	if (!fi.good())
	{
		MFEM_ABORT("Unable to open mesh file \"" << path << "\"!");
	}
	Mesh mesh = Mesh(fi, 1, 1, true);
	mesh.EnsureNodes();
	return mesh;
}

int main(int argc, char* argv[])
{
	// 1. Parse command line options.
	const char* mesh_file = "../data/sphere_tet4.e"; // mesh generated by cubit, saved in netcdf w/o hdf5 enabled
	int order = 1;

	OptionsParser args(argc, argv);
	args.AddOption(&mesh_file, "-m", "--mesh", "Mesh file to use.");
	args.AddOption(&order, "-o", "--order", "Finite element polynomial degree");
	args.ParseCheck();

	// 2. Read the mesh from the given mesh file, and refine once uniformly.
	//Mesh mesh(mesh_file);
	Mesh mesh = LoadMeshNew(mesh_file);
	int dim = mesh.Dimension();

	mesh.UniformRefinement();

	// 3. Define a finite element space on the mesh. Here we use H1 continuous
	//    high-order Lagrange finite elements of the given order.
	H1_FECollection fec(order, mesh.Dimension());
	FiniteElementSpace fespace(&mesh, &fec);
	cout << "Number of unknowns: " << fespace.GetTrueVSize() << endl;

	// 4. Extract the list of all the boundary DOFs. These will be marked as
	//    Dirichlet in order to enforce zero boundary conditions.
	Array<int> boundary_dofs;
	fespace.GetBoundaryTrueDofs(boundary_dofs);

	// 5. Define the solution x as a finite element grid function in fespace. Set
	//    the initial guess to zero, which also sets the boundary conditions.
	GridFunction x(&fespace);
	x = 0.0;

	// 6. Set up the linear form b(.) corresponding to the right-hand side.
	//ConstantCoefficient one(1.0);
	DeltaCoefficient *delta_one; // add point source
	if (mesh.Dimension()==1)
	{
		delta_one = new DeltaCoefficient(0.0, 1.0);
	}
	else if (mesh.Dimension()==2)
	{
		delta_one = new DeltaCoefficient(0.0, 0.0, 1.0);
	}
	else if (mesh.Dimension() == 3)
	{
		delta_one = new DeltaCoefficient(0.0, 0.0, 0.0, 1.0);
	}
	LinearForm b(&fespace);
	//b.AddDomainIntegrator(new DomainLFIntegrator(one));
	b.AddDomainIntegrator(new DomainLFIntegrator(*delta_one));
	b.Assemble();

	// 7. Set up the bilinear form a(.,.) corresponding to the -Delta operator.
	BilinearForm a(&fespace);
	a.AddDomainIntegrator(new DiffusionIntegrator);
	a.Assemble();

	// 8. Form the linear system A X = B. This includes eliminating boundary
	//    conditions, applying AMR constraints, and other transformations.
	SparseMatrix A;
	Vector B, X;
	a.FormLinearSystem(boundary_dofs, x, b, A, X, B);

	// 9. Solve the system using PCG with symmetric Gauss-Seidel preconditioner.
	//GSSmoother M(A);
	//PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#ifndef MFEM_USE_SUITESPARSE
	// Use a simple symmetric Gauss-Seidel preconditioner with PCG.
	GSSmoother M(A);
	PCG(A, M, B, X, 1, 200, 1e-12, 0.0);
#else
	// If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
	UMFPackSolver umf_solver;
	umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
	umf_solver.SetOperator((SparseMatrix&)(A));
	umf_solver.Mult(B, X);
#endif

	// 10. Recover the solution x as a grid function and save to file. The output
	//     can be viewed using GLVis as follows: "glvis -m mesh.mesh -g sol.gf"
	a.RecoverFEMSolution(X, b, x);
	x.Save("sol.gf");
	mesh.Save("mesh.mesh");

	// 11. Save data in the ParaView format
	ParaViewDataCollection paraview_dc("ex0_point_source_debug", &mesh);
	paraview_dc.SetPrefixPath("ParaView");
	paraview_dc.SetLevelsOfDetail(order);
	//paraview_dc.SetCycle(0);
	paraview_dc.SetDataFormat(VTKFormat::BINARY);
	paraview_dc.SetHighOrderOutput(true);
	paraview_dc.SetTime(0.0); // set the time
	paraview_dc.RegisterField("potentail", &x);
	paraview_dc.Save();

	return 0;
}
