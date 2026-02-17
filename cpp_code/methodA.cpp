// File: simExt.cpp
// Purpose: Nab Analysis Simulation software. Extracts teardrop and extracts
// "a" (and maybe "b" at some point)

// C++ includes:
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <math.h>
//#include <cstring>

// This package
//#include "inc/physics.hh"
#include "inc/methodAFit.hh"


// For fitting multiple histograms
#include "Fit/Fitter.h"
#include <Math/Functor.h>


#ifndef __CLING__
int main(int argc, char** argv)
{	
    fit_noE(argc, argv);
    return 0;
}

#else

#include "src/methodAFit.cpp"

int methodA() {	
	std::cout<<"CINT not defined"<<std::endl;
	int argc;
	char** argv;
	fit_noE(argc, argv);
	return 0;
}
#endif

