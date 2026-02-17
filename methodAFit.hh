#ifndef methodAFit_hh
#define methodAFit_hh

// Port of Stefans_Code/fit_noE.cpp (orignal version from Carter Hedinger, UVa)
//#include "physics.hh"

//#include <assert.h>
#include <iostream>
#include <stdio.h>
#include <math.h>
//#include <map>
#include "TMath.h"
//#include <TH2D.h>
#include "TRandom.h"
//#include <TVirtualFitter.h>
//#include <TCanvas.h>
#//include <TStyle.h>
#include "TTime.h"
#include "TSystem.h"

// Other ROOT loads
#include "TFile.h"
#include "TTree.h"
#include "TApplication.h"
#include "TCanvas.h"
#include "TH2D.h"
#include "TStyle.h"
#include "TMarker.h"
#include "TRandom3.h"
#include "TVirtualFitter.h"



// Global variables (in this case, mainly histograms) /
// Define histograms of our dataset
//TH2D* h_data = new TH2D("h_data", "Simulated Data Histogram;E_e [keV];t_p^-2 [s^-2];Number of Events", 
//					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax, 
//					fitOpt::numBinsY, fitOpt::ymin, fitOpt::ymax); // histogram of data

// FUNCTIONS //
double eEspec(double x);
double eECal(double eE, double offset, double gain, double Nonlin);
double EnsureRange(double val, double min, double max);

double DBMapping(double Et2, double dEt2_0, double dEt2_M1, double dEt2_1, double dEt2_2, double dEt2_3, double dEt2_4);
double SBMapping(double L, double Ep, double dEt2_0, double dEt2_M1, double dEt2_1, double dEt2_2, double dEt2_3, double dEt2_4);
double Waleed(double pp2, double cosTh0);

void chi2(int & nPar, double* grad, double &fval, double *par, int iflag);
int fit_noE(int argc, char** argv);

namespace si {
	// For scaling orders of magnitude
	
	// Length units
	constexpr double m  = 1E0;
	constexpr double mm = 1E-3; // Millimeter is base unit
	
	// Timing units
	constexpr double s  = 1E6;
	constexpr double us = 1E0; // Microsecond is base unit
	constexpr double ns = 1E-3; 
	
	// Energy units
	constexpr double MeV = 1E3; 
	constexpr double keV = 1E0; // keV is base unit 
	constexpr double eV  = 1E-3;
	
	// Voltage units 
	constexpr double V = 1E-3; // kV is base unit
}

/*namespace rplot {
	constexpr int nBinsE = 160; // Number of bins in energy histogram
	constexpr int nBinsP = 160; // Number of bins in time of flight histogram
	//const int nBinsE = 80; // Number of bins in energy histogram
	//const int nBinsP = 80; // Number of bins in time of flight histogram


	constexpr double eMin = 1*si::keV;
	constexpr double eMax = 801*si::keV;
	
	constexpr double invTMin = 0/(si::us*si::us);
	constexpr double invTMax = 0.0065/(si::us*si::us);
	
}*/

namespace physics { 
	
	// Physics Constants used here
	constexpr double me = 510999.06;              //mass of electron in eV/c^2
	constexpr double mn = 939565641.8;            //mass of neutron in eV/c^2
	constexpr double delta = 1293331.8;           //mass difference between neutron and proton in eV/c^2
	constexpr double E0 = delta - me; 
	constexpr double mp = mn - delta;             //mass of proton in eV/c^2
	constexpr double pi = TMath::Pi();            //pi (pulled from ROOT)
	constexpr double alpha_SI = 1/137.036;        //fine structure constant
	constexpr double c_SI = 2.99792e8;            //speed of light in m/s

	//constexpr double t2factor = c_SI*c_SI/(mn - delta)/(mn - delta)*1e12; // this is in us^2/(ev/c^2)^2
	constexpr double t2factor = c_SI*c_SI/(mn - delta)/(mn - delta); // this is in s^2/(ev/c^2)^2

	// Physics functions (do as template?)
	double pe(double Ee);
	double pe2(double Ee);
	double pv(double Ee);
	double ppmin(double Ee);
	double ppmax(double Ee);
	double pp2diff(double Ee);
	
	
	double ppmid(double Ee);
	double ppmid2(double Ee);
	double beta(double Ee); 
	double costheta_ev(double E_e, double pp2);
	double gamma_C(double Ee);

} // physics 

// CONSTANTS //
namespace fitOpt {
	
	constexpr double Ep_offset_for_e2 = 1000.0;
	constexpr double cosThetaMin_ = 0;              //minimum cos value (no negative angles)

	constexpr double xmin = 0.01e6; // eV
	constexpr double xmax = 0.81e6; // eV
	constexpr double ymin = 0.0; // s^-2
	constexpr double ymax = 0.0065e12; // s^-2
	//constexpr double ymax = 0.0065; // us^-2
	//constexpr int numBinsX = 160;
	constexpr int numBinsX = 80;
	constexpr int numBinsY = 80;
	constexpr double xBinWidth = (xmax - xmin)/numBinsX;
	constexpr double yBinWidth = (ymax - ymin)/numBinsY;
}

//----------------------------------------------------------------------
// Class Definition
class methodAFit {
	
	public: 
		// Constructor/Destructor
		methodAFit();
		~methodAFit();
		
		// Operator		
		double operator() (double* val, double* par);
		
		// Getters
		double getSpecFierz(int bin, double b);
		//double getEt2_skip() { return this->Et2_skip; } 
		//double gete2_skip()  { return this->e2_skip; } 
		
		// Setters 
		void SetybCorr(double yld1, double yld2) {
			this->ybCorr[0] = yld1;
			this->ybCorr[1] = yld2;
		}
		
		void resetChan(){ 
			for (int i = 0; i < this->nChan; i++) {
				this->chan[i] = 0;
			}
		}
		
		void sethvMapping(double expM1, double exp0, double exp1,
						  double exp2,  double exp3, double exp4) {
			this->hvMapping[0] = expM1;
			this->hvMapping[1] = exp0;
			this->hvMapping[2] = exp1;
			this->hvMapping[3] = exp2;
			this->hvMapping[4] = exp3;
			this->hvMapping[5] = exp4;
			
		}
		
		// External Functions		
		void FillChannels(int i0, double lBnd, double rBnd, 
							double cosMin, double cosMax, double pp2, 
							double intens, double* params);
		void fillHistsDebug(double* params, int* nfitdata);
		void simulateET2SpecMethodA(double* params);
		
		double TOF_MethodA(double cosTh, double pp2, double* par);
				
		double numElectrons(double eMin, double eMax, double b);
		void reconstructEe(double* params);
		
		double getChan(int i) { return this->chan[i]; }
						
		inline double Et2_to_di (double Et2);
		inline double di_to_Et2 (double i);
		
		inline double e2_to_di (double e2);
		inline double di_to_e2 (double i);
		
	private:
		
		// Internal Functions
		double HVCorrection(double pp2);
		
		// Internal Variables
		
		// These are parameters that we're intentionally not setting 
		// for now.
		double* tailStruct;
		double* ybCorr;
		double* detSca;
		double* hvMapping;
		
				
		// These values should be set up to be actually variable...
		// Also maybe not inside of this object?
		
		// Time of Flight (y-axis) parameters
		int Et2_npts = fitOpt::numBinsY; 
		double Et2_step = fitOpt::yBinWidth; //  (fitOpt::ymax)/double(Et2_npts+Et2_skip);
		double Et2_start = fitOpt::ymin; // Et2_step*double(Et2_skip); // start of Et2 histogram (left edge of first bin)
		//int Et2_skip = 0; // TODO Et2_skip and e2_skip don't work right now
		//int Et2_npts = fitOpt::numBinsY - Et2_skip;
		//double Et2_step = (fitOpt::ymax-fitOpt::ymin)/double(Et2_npts+Et2_skip);
		//double Et2_start = Et2_step*double(Et2_skip) + Et2_step/2 + fitOpt::ymin;
		//double Et2_step  = (0.0065e12)/double(Et2_npts + Et2_skip); // Unit: s^-2
		//double Et2_step  = (0.0065)/double(Et2_npts + Et2_skip); // Unit: us^-2
				
		// Electron Energy (x-axis) parameters
		int e2_npts = fitOpt::numBinsX;
		double e2_step = fitOpt::xBinWidth;
		double e2_start = fitOpt::xmin; // (left edge of first bin)
		//int e2_skip = 0;	
		//int e2_npts = (fitOpt::numBinsX - e2_skip);
		//double e2_step = (fitOpt::xmax-fitOpt::xmin)/double(e2_npts+e2_skip);
		//double e2_start = e2_step*double(e2_skip) + e2_step/2 + fitOpt::xmin;
			
		double cosThetaMax = 1.; // Maximum angle (Not sure why this wouldn't always be 1? - special (and maybe outdated) version with e field - SB)
		double rBDet = 1.3/1.8; // Ratio of B-Decay / B-Detector
		
		double detectorHV = 30000.;
		//double eps = 1e-10;

		const int npos = 5; // Number of beam positions to smear over
		
		static constexpr int nChan = fitOpt::numBinsX*fitOpt::numBinsY;
		double chan[nChan]; 
		
		
		
		
		// Arrays for holding the eE distribution function
		static constexpr int lenSpec = 999;  
		//static const int lenSpec = 499;   
		//double* eESpec_raw;
		//double* eESpec_me;
		
		double eESpec_raw[lenSpec+1];
		double eESpec_me[lenSpec+1];
		
};

// TODO at some point we should create a "Calibration" object to take some
// of that functionality away from other things 
// (and make us not have 20-something fit parameters in the function...)
class energy2ADC {
				
	public:
		// Constructor/Destructor
		energy2ADC();
		~energy2ADC(); 
		
		// Operator
		double calEe(double* val, double* par);
		
				
	private:
		
		double e2start, e2_step;
		
		// Private variables for scaling
		double gainSca = 1;
		double nonlinSca = 1e-6;
		
};

extern "C" {
	methodAFit* methodAFit_new();//{ return new methodAFit(); } 
	void methodASpec(methodAFit* methodA, double* params);// { 
		
}

#endif
