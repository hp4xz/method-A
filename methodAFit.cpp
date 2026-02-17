#ifndef methodAFit_cpp
#define methodAFit_cpp

// Port of Stefans_Code/fit_noE.cpp (Orignal version from Carter Hedinger, UVa)

// #include <excpt.h>
#include "../inc/methodAFit.hh"
//#include "PhysicsConstants.hh"

using namespace std; // TODO REMOVE
// GLOBAL VARIABLES // 
// Global variables used to store fitting parameters
int g_nFillChan, g_nFillChan2, g_nSimulateET2Spec;
methodAFit* g_fitObject = new methodAFit; // Initialize a global fitting object
const bool FitRegionIncludesEdges = true;
const bool UseHVCorrection = false;
// #define FillChannelsWithoutRecursion

TH2D* h_data = new TH2D("h_data", "Real Data Histogram;E_e [keV];t_p^-2 [s^-2];Number of Events", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax, 
					fitOpt::numBinsY, fitOpt::ymin, fitOpt::ymax); // histogram of data
TH2D* h_fit2D = new TH2D("h_fit2D", "Fit Histogram;E_e [keV];t_p^-2 [s^-2];Number of Events", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax, 
					fitOpt::numBinsY, fitOpt::ymin, fitOpt::ymax); // histogram of fit
TH2D* h_residual = new TH2D("h_residual", "Residual Histogram;E_e [eV];t_p^-2 [s^-2];Number of Events", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax, 
					fitOpt::numBinsY, fitOpt::ymin, fitOpt::ymax); // histogram of residuals
TH1D* h_low_outer= new TH1D("h_low_outer", "Lower bound of outer range;E_e [eV];t_p^-2 [s^-2]", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax); // lower bound of outer range (incl. edges)
TH1D* h_low_inner= new TH1D("h_low_inner", "Lower bound of inner range;E_e [eV];t_p^-2 [s^-2]", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax); // lower bound of inner range (incl. edges)
TH1D* h_up_outer= new TH1D("h_up_outer", "Upper bound of outer range;E_e [eV];t_p^-2 [s^-2]", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax); // upper bound of outer range (incl. edges)
TH1D* h_up_inner= new TH1D("h_up_inner", "Upper bound of inner range;E_e [eV];t_p^-2 [s^-2]", 
					fitOpt::numBinsX, fitOpt::xmin, fitOpt::xmax); // upper bound of inner range (incl. edges)


namespace physics { 
	
	// Physics functions (do as template?)
	double pe(double Ee) { return std::sqrt((Ee + me)*(Ee + me) - (me*me)); }
	double pe2(double Ee) { return (Ee + me)*(Ee + me) - (me*me); }
	double pv(double Ee) { return delta - me - Ee; }
	double ppmin(double Ee) { return pe(Ee) - pv(Ee); } // 1e12 to convert from s^-1 to us^-1
	double ppmax(double Ee) { return pe(Ee) + pv(Ee); }
	double pp2diff(double Ee) { return 4*pe(Ee)*pv(Ee); } // ppmax*ppmax - ppmin*ppmin
		
	double ppmid(double Ee) { return std::sqrt(pe2(Ee) + pv(Ee)*pv(Ee)); }
	double ppmid2(double Ee) { return pe2(Ee) + pv(Ee)*pv(Ee); } 
	double beta(double Ee) {
		if (Ee > 0) {
			return pe(Ee)/(me + Ee);
		} else {
			return 0;
		}
	}

	double costheta_ev(double E_e, double pp2) {
		return (pp2 - pe2(E_e) - (pv(E_e)*pv(E_e)))/(2*pe(E_e)*pv(E_e));
	}
	double gamma_C(double Ee) {
		return 1/std::sqrt(1 - beta(Ee)*beta(Ee));
	}

} // physics 

//TODO: Should be compiled with /EHa compiler option to catch numerical exceptions.

// FUNCTION DEFINITIONS //
double eEspec(double x) {
	if ((0 < x) && (x < physics::E0)) {
		double betaVal = physics::beta(x);
		//CC = 1+physics::alpha_SI*physics::pi/betaVal
		//     +(physics::alpha_SI*physics::alpha_SI)*(11/4-0.5772-TMath::Log(2*betaVal*(me_eV+x)*0.01/4/me_eV+(physics::pi/betaVal)*(physics::pi/betaVal)/3));
		double corr = (2*physics::pi*physics::alpha_SI/betaVal)
					/(1 - exp(-2*physics::pi*physics::alpha_SI/betaVal));
		return (physics::E0 - x)*(physics::E0 - x)*physics::pe(x)*(x + physics::me)*corr;
	} else {
		return 0;
	}
}

double EnsureRange(double val, double min, double max) {
  if (val < min) { return min; }
  if (val > max) { return max; }
  return val;
}

double eECal(double e2, double offset, double gain, double Nonlin) {
	// This is assuming we have some imperfect calibration for our detector
	double energyADC = offset + gain*e2 + (Nonlin/1e6)*gain*gain*e2*e2;
	
	if (energyADC < 0) { return 0; } 
	else if (energyADC > physics::E0) { return 0; } 
	else { return energyADC; }
}

// These are dumb mapping things.
// I'm going to make these maps work but they're not called here
double DBMapping(double* val, double* parHV) {
	
	double Et2 = val[0];
	return Et2  + 1e-6*parHV[0]/Et2 
				+ 1e-3*parHV[1] 
				+ 1e-3*parHV[2]*Et2
				+ 1.*  parHV[3]*Et2*Et2
				+ 1e+2*parHV[4]*Et2*Et2*Et2
				+ 1e+4*parHV[5]*Et2*Et2*Et2*Et2;
}

double SBMapping(double L, double pp2, double* parHV) {
		
	return L + 1e+2* parHV[0]/pp2
			 + 1e-3* parHV[1]
			 + 1e-3* parHV[2]*pp2
			 + 1e-6* parHV[3]*pp2*pp2
			 + 1e-8* parHV[4]*pp2*pp2*pp2
			 + 1e-11*parHV[5]*pp2*pp2*pp2*pp2;
}

double Waleed(double pp2, double cosTh0) {
	// This is a parametrization of detector efficiency
	
	const double c1 = -0.0251;
	const double c2 = 1.09;
	const double c3 = 1.65;
	const double c4 = -31.67;
	const double sigma = 1.61;
	const double DetHV = -30000;
	const double Bdet= 1.3;
	const double B0 = 1.8;

	double T0_keV = pp2/2/physics::mn/1000;
	double TDet_keV = T0_keV - DetHV/1000;
	double cosThDet = sqrt(1 - Bdet/B0*(T0_keV/TDet_keV)*(1 - cosTh0*cosTh0));
	double ThDet = TMath::ACos(cosThDet)*360./physics::pi; // in degrees
	double f_BS = (c1*TDet_keV + c2)
					*(1-c3*(sqrt(2/physics::pi)/sigma)*exp(-(ThDet*ThDet/sigma/sigma)/2) 
					+ (ThDet*ThDet/c4/c4));
	
	return f_BS;
}

//----------------------------------------------------------------------
// MethodAFit Constructor and Destructor
//----------------------------------------------------------------------
methodAFit::methodAFit() {
	
	// TODO make these correction factors work better (e.g. as an object)
	// Set the correction factors to 0.
	
	this->ybCorr = new double[2] {0., 0.,};
	this->tailStruct = new double[3] {0.01,7E-5,0.13};
	this->hvMapping = new double[6] {0.,0.,0.,0.,0.,0.};
	
		
	// Internal array containing the list of data that we want	
	this->resetChan();

	// Initialize the unweighted energy spectrum 
	//this->eESpec_raw = new double[this->lenSpec + 1];
	//this->eESpec_me  = new double[this->lenSpec + 1];
	this->eESpec_raw[0] = 0.;
	this->eESpec_me[0]  = 0.;
	
	// Now loop through and generate the initial spectrum
	double temp1 = 0.;
	double temp2 = 0.;
	for (int ie = 1; ie <= this->lenSpec; ie++) {
		
		double availE = physics::delta - physics::me;
		double midVal  = (double(ie)-0.5)/(this->lenSpec)*availE;
		double specVal = eEspec(midVal);
				
		temp1 += specVal;
		this->eESpec_raw[ie] = temp1;
		
		temp2 += specVal*physics::me/(physics::me + midVal);
		this->eESpec_me[ie] = temp2;
	}
	// We then reweight this to normalize to the endpoint
	double specNorm = this->eESpec_raw[this->lenSpec];
	for (int ie = 0; ie <= this->lenSpec; ie++) {
		this->eESpec_raw[ie] /= specNorm;
		this->eESpec_me[ie]  /= specNorm;
	}
	
	
}

methodAFit::~methodAFit() {
	
	//delete[] chan;
	//delete[] eESpec_raw;
	//delete[] eESpec_me;
	
	
	
	delete[] tailStruct;
	delete[] ybCorr;
	
	delete[] detSca;
	delete[] hvMapping;
		
}

//----------------------------------------------------------------------
// MethodAFit Methods
//----------------------------------------------------------------------
double methodAFit::TOF_MethodA(double cosTh, double pp2, double* par) {
	// This calculates the time of flight assuming a taylor expansion.
	// As an operator with arrays rather than an object
	// Takes two inputs: cosTheta and pp2 (which is used if we want to 
	//     correct for proton momentum later
	
	// Our "Parameters" that we're fitting for
	double A_cosThetaMin = par[0];
	//double tmp_cosThetaMin = par[1];
	//double tmp_cosThetaMax = par[2]; // Doesn't do anything here
	double L_zDV = par[1];
	double A_alpha  = par[2];
	double A_beta   = par[3];
	double A_gamma  = par[4];
	double A_eta    = par[5];
	
	double cosmid = (1 + A_cosThetaMin)/2;
	// We get zeros if we're right at the minimum angle
	
	if ((cosTh <= A_cosThetaMin) || (A_cosThetaMin >= 1) || (cosTh > 1)) {
		cout << "TOF_MethodA error: cosTh (" << cosTh << ") <= A_cosThetaMin (" << A_cosThetaMin << ", diff: "<< cosTh - A_cosThetaMin <<")" << endl;
		cosTh += 1e-6;
	}
	if ((L_zDV < 4) || (L_zDV > 6) || (A_alpha < -5) || (A_alpha > 5) || (A_beta < -1)
		|| (A_beta > 1) || (A_gamma < -5) || (A_gamma > 5) || (A_eta < -1) || (A_eta > 1)) { 
		cout << "TOF_MethodA error: Illegal calling parameters (L_zDV = " << L_zDV << ", alpha = " << A_alpha << ", beta = " << A_beta << ", gamma = " << A_gamma << ", eta= " << A_eta << endl;
		return 5.;
	}
	
	double cosTSub = (cosTh - cosmid);
	
	
	// And the result
	double result;
	if (UseHVCorrection) { 
		double hvCorr = this->HVCorrection(pp2/1e12);
		result = hvCorr*(L_zDV
				- A_eta*TMath::Log((cosTh - A_cosThetaMin)/(1 - A_cosThetaMin)) 
				- A_alpha*cosTSub 
				+ A_beta *cosTSub*cosTSub
				- A_gamma*cosTSub*cosTSub*cosTSub);
	} else { 
		result = L_zDV
				- A_eta*TMath::Log((cosTh - A_cosThetaMin)/(1 - A_cosThetaMin)) 
				- A_alpha*cosTSub 
				+ A_beta *cosTSub*cosTSub
				- A_gamma*cosTSub*cosTSub*cosTSub; 
	}
	
	//double Leff = SBMapping(LM5 + 5.0 - zDV, 
	//					pp2/(2*physics::mp), A_dEt2_0, A_dEt2_M1, 
	//					A_dEt2_1, A_dEt2_2, A_dEt2_3, A_dEt2_4);
	//double SBbeta = A_beta; // +9e-5*pp2/(2*physics::mp);
	//result = Leff*(1 - A_alpha*(cosTh - cosmid) 
	//				+ SBbeta*cosTSub*cosTSub
	//				- A_gamma*cosTSub*cosTSub*cosTSub)
	//				- A_eta*TMath::Log((cosTh - A_costhetamin)/(1 - A_costhetamin));
	
	// Corrections to 1/t_p^2
	// if (this->CorrectTof()) { 
	/*if ((para.A_dEt2_0 != 0) || (para.A_dEt2_M1 != 0) 
			|| (para.A_dEt2_1 != 0) || (para.A_dEt2_2 != 0) 
			|| (para.A_dEt2_3 != 0) || (para.A_dEt2_4 != 0) 
			|| (para.A_dt != 0)) {
		double Et2corr = DBMapping(1e-12*physics::t2factor*pp2/(result*result), 
									para.A_dEt2_0, para.A_dEt2_M1, para.A_dEt2_1, 
									para.A_dEt2_2, para.A_dEt2_3, para.A_dEt2_4);
		if (Et2corr > 1e-4) {
			Et2corr = 1/((1/sqrt(Et2corr) + para.A_dt/1000)*(1/sqrt(Et2corr) + para.A_dt/1000));
			if (Et2corr > 1e-4) {
				result = 1/sqrt(Et2corr/(1e-12*physics::t2factor*pp2));
			}
		}
		Et2corr = 1e-4*exp(Et2corr - 1/1e4);
		result = 1/sqrt(Et2corr/(1e-12*physics::t2factor*pp2));
	}*/
  return result;

}

double methodAFit::HVCorrection(double pp2) {
	// TODO figure out what the coefficients for HV mapping should be.
	//std::cout << pp2 << std::endl;
	return 1 + 1e-2*this->hvMapping[0]/pp2
			 + 1e-3*this->hvMapping[1]
			 + 1e-3*this->hvMapping[2]*pp2
			 + 1e-6*this->hvMapping[3]*pp2*pp2
			 + 1e-8*this->hvMapping[4]*pp2*pp2*pp2
			 + 1e-11*this->hvMapping[5]*pp2*pp2*pp2*pp2;
	
}

inline double methodAFit::Et2_to_di (double Et2) {
  return( Et2 - this->Et2_start)/this->Et2_step;
  // note: return value 0....0.9999 is in bin 0;
}

inline double methodAFit::di_to_Et2 (double i) {
  return i*this->Et2_step+this->Et2_start;
  // note: first bin has bin number zero. Return value is left edge, not bin center
}

inline double methodAFit::e2_to_di (double e2) {
  return( e2 - this->e2_start)/this->e2_step;
  // note: return value 0....0.9999 is in bin 0;
}

inline double methodAFit::di_to_e2 (double i) {
  return i*this->e2_step+this->e2_start;
  // note: first bin has bin number zero. Return value is left edge, not bin center
}

#ifdef FillChannelsWithoutRecursion
constexpr int FIFOSize = 5*fitOpt::numBinsY;
double lBnd_FIFO[FIFOSize], rBnd_FIFO[FIFOSize], cosMin_FIFO[FIFOSize], cosMax_FIFO[FIFOSize];

void methodAFit::FillChannels(int i0, double lBnd, double rBnd, 
								double cosMin, double cosMax, 
								double pp2, double intens, double* paramsA) {
	// On the first pass of this, let's go and calculate the boundaries.	
	if (cosMin == paramsA[0]) { cout << "Error in FillChannels: " << cosMin << " " << cosMax << endl; }
	
	// define four FIFO stack for cosTheta0 values and ET2 values
	int it_FIFO= 0; // number of last FIFO element (common to all four)
	int i_FIFO= -1; // number of FIFO element being worked on (common to all four)
	
	// fill first FIFO element
	cosMin_FIFO[it_FIFO]= cosMin;
	cosMax_FIFO[it_FIFO]= cosMax;
	// compute first right bound channel			
	double tp = this->TOF_MethodA(cosMax,pp2,paramsA);
	if ((tp < 4) || (tp > 6)) {
		cout << "FillChannels error 1" << endl;
		return;
	}
	double Et2 = physics::t2factor*pp2/(tp*tp);
	rBnd_FIFO[it_FIFO] = Et2_to_di( Et2);		
	// compute first left bound channel			
	tp = this->TOF_MethodA(cosMin,pp2,paramsA);
	if ((tp < 4) || (tp > 8)) {
		cout << "FillChannels error 2 (tp: " << tp << ")" << endl;
		return;
	}
	Et2 = physics::t2factor*pp2/(tp*tp);
	lBnd_FIFO[it_FIFO] = Et2_to_di( Et2);
		
	// now fill the intervals in FIFO or continue to split until they are all taken care of
	do {
		// We care about both double and int versions of the boundaries
		i_FIFO++;
		int ilBnd = TMath::FloorNint(lBnd_FIFO[i_FIFO]);
		int irBnd = TMath::FloorNint(rBnd_FIFO[i_FIFO]);
	
		// Error handling
		if (rBnd_FIFO[i_FIFO] <= lBnd_FIFO[i_FIFO]) 		{
			//std::cout << "lBnd (" << cosMin_FIFO[i_FIFO] << ", " << lBnd_FIFO[i_FIFO] << ") > rBnd (" << cosMax_FIFO[i_FIFO]<< ", " << rBnd_FIFO[i_FIFO] << ")"
			//<< " [alpha = " << paramsA[2] << ", beta = " << paramsA[3] << "]" <<std::endl; 
			return; } // Right has to be bigger than left
		if (ilBnd >= this->Et2_npts) {
			continue;
		} // Left bound is too big
		if (irBnd < 0) {
			continue;
		} // Right bound is too small
	
		if ((rBnd_FIFO[i_FIFO] - lBnd_FIFO[i_FIFO] > 0.05) && (ilBnd != irBnd)) {
		
			// We now divide this angle to extrapolate between them
			double cosMid = (3.*cosMin_FIFO[i_FIFO] + cosMax_FIFO[i_FIFO])/4.;
				
			// Calculate the time of flight, and then re-run this function
			tp = this->TOF_MethodA(cosMid,pp2,paramsA);
			if (tp < 4) {
				cout << "FillChannels error 3" << endl;
				return;
			}
			Et2 = physics::t2factor*pp2/(tp*tp);
			double mBnd = Et2_to_di( Et2);	;
			
		    // Now we add in a new boundary
			if (it_FIFO >= FIFOSize-2) {
				cout << "FillChannels: FIFO to small" << endl;
				return;
			}
			it_FIFO++;
			lBnd_FIFO[it_FIFO]= lBnd_FIFO[i_FIFO];
			rBnd_FIFO[it_FIFO]= mBnd;
			cosMin_FIFO[it_FIFO]= cosMin_FIFO[i_FIFO];
			cosMax_FIFO[it_FIFO]= cosMid;
			it_FIFO++;
			lBnd_FIFO[it_FIFO]= mBnd;
			rBnd_FIFO[it_FIFO]= rBnd_FIFO[i_FIFO];
			cosMin_FIFO[it_FIFO]= cosMid;
			cosMax_FIFO[it_FIFO]= cosMax_FIFO[i_FIFO];
			/* for (int i=i_FIFO+1; i<= it_FIFO; i++) {
				cout << "(" << i << ": "<< lBnd_FIFO[i] << " " << cosMin_FIFO[i] << ", " << rBnd_FIFO[i] << " " << cosMax_FIFO[i] << "), " << endl;
			}
			cout << endl;
			*/
		} else {
	
			// Now we can fill the chan array (the 2D histogram of Ee and Et2)
			g_nFillChan2++;
			double dcos = cosMax_FIFO[i_FIFO] - cosMin_FIFO[i_FIFO];
		
			// Two ways of scaling the intensity (smearing the pp2)
			// TODO Make this its own thing (so we can account for E fields)
			double DetCorr = 1;
			//if (this->detSca != NULL) { 
			//	bool debugBOOL = true;
			//	if (debugBOOL) { // This is "Waleed's Equation"
			//		DetCorr = 1  - detSca[0]*Waleed(pp2, (this->cosMax + cosMinTmp)/2);
			//	} else { // This is the "usual definition for det_c1"
			//		double cosDet = sqrt(1 - this->rBDet*(1 - cosMid*cosMid)
			//					/(1 - 2*physics::mp*this->detectorHV/pp2)); // wrong!! (/2 -> /4)
			//		DetCorr= 1 + detSca[0]*1e-6*(pp2/2/physics::mp)/1000-detSca[1]*1e-2/cosDet;
			//	}
			//}
		
			// Scale the intensity and calculate the bin size
			double intensTmp2 = intens*DetCorr; // intensity gets scaled by the extended TOF
			double dt2 = 1./(rBnd_FIFO[i_FIFO] - lBnd_FIFO[i_FIFO]);
		
			if (ilBnd == irBnd) { // If we just have one point
				double dx = rBnd_FIFO[i_FIFO] - lBnd_FIFO[i_FIFO];
				if ((0 <= ilBnd) && (ilBnd <= this->Et2_npts-1)) {
					// Do a correction for yield scaling (?)
					//double et2 = 1e-12*(((double)ilBnd)*this->Et2_step + this->Et2_start);
					double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));	
					if ((ilBnd < 0) || (ilBnd >= (int)this->Et2_npts) || (intensTmp2*dt2*dx*dcos*yldScale == 0)) {
						cout << "Error 1 in Fillchan logic!" << endl;
					}
				
					// Add incremental amount to this channel
					this->chan[i0+ilBnd] += intensTmp2*dt2*dx*dcos*yldScale;
				} else {
					cout << "FillChannels error 5" << endl;
					return;
				}
			} else { // Many points in the range
				if ((0 <= ilBnd) && (ilBnd <= this->Et2_npts-1)) {
					double dx = (double)(ilBnd+1) - lBnd_FIFO[i_FIFO];
					// Correction for yield scaling
					//double et2 = 1e-12*(((double)ilBnd)*this->Et2_step + this->Et2_start);
					double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));
					if ((ilBnd < 0) || (ilBnd >= (int)this->Et2_npts) || (intensTmp2*dt2*dx*dcos*yldScale == 0)) {
						cout << "Error 2 in Fillchan logic!" << endl;
					}
				
					// Add incremental amount to this channel
					this->chan[i0+ilBnd] += intensTmp2*dt2*dx*dcos*yldScale;
				}
				for (int it2 = max( ilBnd+1, 0); it2 <= min(irBnd-1, (int)(this->Et2_npts)-1); it2	++) {
					//double et2 = 1e-12*(((double)it2)*this->Et2_step + this->Et2_start);
					double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + 	this->ybCorr[1]*1e-9/(et2*et2));
				
					if ((it2 < 0) || (it2 >= (int)this->Et2_npts)) {
						cout << "Error 3 in Fillchan logic!" << endl;
					}
					// Add incremental amount to this channel
					this->chan[i0+it2] += intensTmp2*dt2*dcos*yldScale;
				}
				if (irBnd < (int)this->Et2_npts) {
					double dx = rBnd_FIFO[i_FIFO] - (double)irBnd;
					// Correction for yield scaling
					//double et2 = 1e-12*(((double)irBnd)*this->Et2_s	tep + this->Et2_start);
					double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));
					if ((irBnd < 0) || (irBnd >= (int)this->Et2_npts)) {
						cout << "Error 4 in Fillchan logic!" << endl;
					}
				
					// Add incremental amount to this channel
					this->chan[i0+irBnd] += intensTmp2*dt2*dx*dcos*yldScale;
				}
			}	
		}
	} while (i_FIFO < it_FIFO);
}
#endif

#ifndef FillChannelsWithoutRecursion
void methodAFit::FillChannels(int i0, double lBnd, double rBnd, 
								double cosMin, double cosMax, 
								double pp2, double intens, double* paramsA) {
	// On the first pass of this, let's go and calculate the boundaries.	
	if (cosMin == paramsA[0]) { cout << "Error in FillChannels: " << cosMin << " " << cosMax << endl; }
	if (rBnd < 0) { // Replacing NULL with negative
				
		double tIntMax = this->TOF_MethodA(cosMax,pp2,paramsA);
		if (tIntMax < 4) {
			cout << "FillChannels error 1" << endl;
			return;
		}
		double tof2Max = physics::t2factor*pp2/(tIntMax*tIntMax);
	
		rBnd = Et2_to_di( tof2Max);		
	}
	if (lBnd < 0) {
		double tIntMin = this->TOF_MethodA(cosMin,pp2,paramsA);
		if (tIntMin < 4) {
			cout << "FillChannels error 2" << endl;
			return;
		}
		double tof2Min = physics::t2factor*pp2/(tIntMin*tIntMin);
	
		lBnd = Et2_to_di( tof2Min);
	}
	// We care about both double and int versions of the boundaries
	int ilBnd = TMath::FloorNint(lBnd);
	int irBnd = TMath::FloorNint(rBnd);
	
	// Error handling
	if (rBnd <= lBnd) 		{
	//	std::cout << "lBnd (" << cosMin << ", " << lBnd << ") > rBnd (" << cosMax<< ", " << rBnd << ")"
	//		<< " [alpha = " << paramsA[2] << ", beta = " << paramsA[3] << "]" <<std::endl; 
		return; 
	} // Right has to be bigger than left
	if (ilBnd >= this->Et2_npts) 	{
	//	std::cout << "lBnd (" << cosMin << ", " << lBnd << ") out of range [alpha = " << paramsA[2] << ", beta = " << paramsA[3] << "]" <<std::endl; 
		 return; 
	} // Left bound is too big
	if (irBnd < 0) 			{
	//	std::cout << "rBnd (" << cosMax << ", " << rBnd << ") out of range [alpha = " << paramsA[2] << ", beta = " << paramsA[3] << "]" <<std::endl;  
		return; 
	} // Right bound is too big
	
	// This is recursion. I'd rather see this as a "while" loop but w/e
	if ((rBnd - lBnd > 0.05) && !(ilBnd == irBnd)) {
		// We now divide this angle to extrapolate between them
		double cosMid = (3.*cosMin + cosMax)/4.;
				
		// Calculate the time of flight, and then re-run this function
		double tInt = this->TOF_MethodA(cosMid,pp2,paramsA);				
		if (tInt < 4) {
			cout << "FillChannels error 3" << endl;
			return;
		}		
		double tof2 = physics::t2factor*pp2/(tInt*tInt);
		
		// Now we add in a new boundary
		double mBnd = Et2_to_di( tof2);	;
		
		// Finally, let's rerun this!
		this->FillChannels(i0,lBnd,mBnd,cosMin,cosMid,pp2,intens,paramsA);
		this->FillChannels(i0,mBnd,rBnd,cosMid,cosMax,pp2,intens,paramsA);
		
	} else {
	
		// We're not continuing to loop here, we just fill this in
		//double cosMid = (cosMax + cosMin) / 2;
		g_nFillChan2++;
		double dcos = cosMax - cosMin;
		
		// Two ways of scaling the intensity (smearing the pp2)
		// TODO Make this its own thing (so we can account for E fields)
		double DetCorr = 1;
		//if (this->detSca != NULL) { 
		//	bool debugBOOL = true;
		//	if (debugBOOL) { // This is "Waleed's Equation"
		//		DetCorr = 1  - detSca[0]*Waleed(pp2, (this->cosMax + cosMinTmp)/2);
		//	} else { // This is the "usual definition for det_c1"
		//		double cosDet = sqrt(1 - this->rBDet*(1 - cosMid*cosMid)
		//					/(1 - 2*physics::mp*this->detectorHV/pp2)); // wrong!! (/2 -> /4)
		//		DetCorr= 1 + detSca[0]*1e-6*(pp2/2/physics::mp)/1000-detSca[1]*1e-2/cosDet;
		//	}
		//}
		
		// Scale the intensity and calculate the bin size
		double intensTmp = intens*DetCorr; // intensity gets scaled by the extended TOF
		double dt2 = 1./(rBnd - lBnd);
		
		if (ilBnd == irBnd) { // If we just have one point
			double dx = rBnd - lBnd;
			if ((0 <= ilBnd) && (ilBnd <= this->Et2_npts-1)) {
				// Do a correction for yield scaling (?)
				//double et2 = 1e-12*(((double)ilBnd)*this->Et2_step + this->Et2_start);
				double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));	
				if ((ilBnd < 0) || (ilBnd >= (int)this->Et2_npts)) {
					cout << "Error 1 in Fillchan logic!" << endl;
				}
				
				// Add incremental amount to this channel
				this->chan[i0+ilBnd] += intensTmp*dt2*dx*dcos*yldScale;
			}
		} else { // Many points in the range
			if ((0 <= ilBnd) && (ilBnd <= this->Et2_npts-1)) {
				double dx = (double)(ilBnd+1) - lBnd;
				// Correction for yield scaling
				//double et2 = 1e-12*(((double)ilBnd)*this->Et2_step + this->Et2_start);
				double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));
				if ((ilBnd < 0) || (ilBnd >= (int)this->Et2_npts)) {
					cout << "Error 2 in Fillchan logic!" << endl;
				}
				
				// Add incremental amount to this channel
				this->chan[i0+ilBnd] += intensTmp*dt2*dx*dcos*yldScale;
			}
			for (int it2 = max( ilBnd+1,0); it2 <= min(irBnd-1, (int)(this->Et2_npts)-1); it2++) {
				//double et2 = 1e-12*(((double)it2)*this->Et2_step + this->Et2_start);
				double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));
				if ((it2 < 0) || (it2 >= (int)this->Et2_npts)) {
					cout << "Error 3 in Fillchan logic!" << endl;
				}
				
				// Add incremental amount to this channel
				this->chan[i0+it2] += intensTmp*dt2*dcos*yldScale;
			}
			if (irBnd < (int)this->Et2_npts) {
				double dx = rBnd - (double)irBnd;
				// Correction for yield scaling
				//double et2 = 1e-12*(((double)irBnd)*this->Et2_step + this->Et2_start);
				double yldScale = 1.;// + (this->ybCorr[0]*1e-6/et2 + this->ybCorr[1]*1e-9/(et2*et2));
				if ((irBnd < 0) || (irBnd >= (int)this->Et2_npts)) {
					cout << "Error 4 in Fillchan logic!" << endl;
				}
								
				// Add incremental amount to this channel
				this->chan[i0+irBnd] += intensTmp*dt2*dx*dcos*yldScale;
			}
		}
	}
}
#endif

double methodAFit::getSpecFierz(int bin, double b) {
	
	// This function gets the value of the spectrum with the Fierz correction
	if ((bin < 0) || (bin > this->lenSpec)) {
		cout << "Ee spectrum out of range (bin " << bin << " )" << endl;
		return 0;
	}
	return this->eESpec_raw[bin] + b*(this->eESpec_me[bin]);
}

double methodAFit::numElectrons(double eMin, double eMax, double b) {
	
	// How many electrons to we get between two bounds
	double availE = physics::delta - physics::me;
	double eStep = availE/((double)(this->lenSpec)); // Total energy available
			
	// Return 0 if we're out of range
	if (eMin > availE) { return 0;}
    if (eStep <= 0) { cout << "numElectrons logic error" << endl; }   
    
	// Find the bins we're going to extrapolate between  
	int binL = EnsureRange(TMath::FloorNint(eMin/eStep-0.5), 0, this->lenSpec);
	int binR = EnsureRange(TMath::FloorNint(eMax/eStep-0.5), 0, this->lenSpec);
		
	// Extrapolation is better if we're averaging the slope on either side
	double resultL = ((eMin/eStep)-binL)*(this->getSpecFierz(binL+1,b) - this->getSpecFierz(binL,b)) 
						+ this->getSpecFierz(binL,b);
	double resultR = ((eMax/eStep)-binR)*(this->getSpecFierz(binR+1,b) - this->getSpecFierz(binR,b)) 
						+ this->getSpecFierz(binR,b);

	// Return the result (scaled to the endpoint)
	if (resultR > resultL) {
		
		double result = resultR - resultL;
		return result / this->getSpecFierz(this->lenSpec,b);
	} else {
		return 0;
	}
}

void methodAFit::simulateET2SpecMethodA(double* params) {
	// This fills the channels array given our parameters

	// Parameters we want to find
	double a_ev = params[0];
	double b_F  = params[1];
	//double intens = 10*params[2];//std::pow(10,params[2]);
	//double intens = 1;
	double intens = std::pow(10,params[2]);
	
	// Nuisance parameters (Method A in general) -- TOF mapping
	double A_cosThetaMin = params[3];
	double A_L = params[4] + 5.;
	double A_alpha = params[5];
	double A_beta  = params[6];
	double A_gamma = params[7];
	double A_eta   = params[8];
	
	// Pass a container for this to put into the object
	double paramsATmp[6] = { A_cosThetaMin,
							A_L,
							A_alpha,
							A_beta,
							A_gamma,
							A_eta };
							
	// Nuisance parameters (In this function loops) -- beam profile
	double z0_center = params[9];
	double z0_width  = params[10];
	
	// Nuisance parameters -- bremsstrahlung
	this->tailStruct[0] = params[11];
	this->tailStruct[1] = params[12]*1E-4;
	this->tailStruct[2] = params[13];
	
	// TODO find what order of magnitude these things should be
	this->sethvMapping(params[14],
					   params[15],
					   params[16],
					   params[17],
					   params[18],
					   params[19]);
	
	// Nuisance parameters -- energy reconstruction
	// These might not be needed given a good source calibration
	double calEe   = params[20];
	double EeNonLinearity = params[21];
	double calibration[2] = {calEe,
							 EeNonLinearity};
	double sigmaEe_keV = params[22];
	
	
	int n_dE=1; // Number of electron energies within one electron energy bin to loop over
	if (sigmaEe_keV > 0) {
		n_dE= TMath::FloorNint(e2_step/1000./sigmaEe_keV);
		if (n_dE < 1)	
			n_dE = 1;
	}
	
	// This function works via nested loops where we smear various parameters
	double norm_e2 = 0;
	for (int ie = 0; ie < this->e2_npts; ie++) { 
		// Loop Through electron Energy
		int ioffset = ie*((int)this->Et2_npts);
		
		for (int ide = 0; ide < n_dE; ide++) { 
			// TODO: Allow to Smear Energy. 
			// At one point they did a Gaussian smear, in the final DELPHI version a rectangular smear.
			// Here, no smear.
			//double offsetTmp = this->e2_start - this->e2_step/2.;
			double tmpCal = calEe*this->e2_step;
			
			// Central value of the energy
			double eE      = eECal((double)ie + ((double)ide+0.5)/n_dE, this->e2_start, tmpCal, EeNonLinearity);
			
			// Get the probability of this energy
			double eMinTmp = eECal((double)ie + ((double)ide)/n_dE, this->e2_start, tmpCal, EeNonLinearity);
			double eMaxTmp = eECal((double)ie + ((double)ide+1.0)/n_dE, this->e2_start, tmpCal, EeNonLinearity);
			double e2prob = this->numElectrons(eMinTmp, eMaxTmp, b_F);
	
			norm_e2 += e2prob; 
			
			// ---------------------------------------------------
			// Find the min/max of the momenta			
			// There's a many-parameter mapping of the Electric Field correction
						
			// We have to smear the proton momentum
			//double pp2diff = physics::pp2diff(eE);
			double pp2max = physics::ppmax(eE)*physics::ppmax(eE);
			double pp2min = physics::ppmin(eE)*physics::ppmin(eE);
			double pp2mid = physics::ppmid2(eE);      
			double pp2denom = 2*(eE + physics::me)*(physics::delta - physics::me - eE);
			
			//int npp = 5*Round(physics::t2factor*pp2diff/((A_LM5 + 5.)*(A_LM5  + 5.))/this->Et2_step);
			int npp = 5*TMath::FloorNint(physics::t2factor*(pp2max-pp2min)
											/(A_L*A_L)/this->Et2_step + 0.5);
			if (npp > 1000) { 
				npp = 1000;
			}
			// Smear the neutron beam
			for (int ipos = 1; ipos <= this->npos; ipos++) {
				
				double tmpPosZ = ((double)ipos - 0.5)/(double)npos - 0.5;
				double zTmp = z0_center + z0_width*tmpPosZ;
				
				// Modification for variable path length
				paramsATmp[1] = A_L - zTmp;
					
				// Smear proton momentum
				for (int ipp = 0; ipp < npp; ipp++) { 
					
					//double pp2 = pp2min + pp2diff*(ipp + 0.5)/npp;
					double pp2 = pp2min + (pp2max-pp2min)*((double)ipp + 0.5)/npp; // Stefan inserts "double; stupid C++ error
					//double dpp2 = 1. / ((double)npp);
					
					// TODO: correction of cosmin if we have electric field between DV and filter has not been translated from DELPHI
				
					// Probability of seeing this momentum 
					double P_p2 = (1 + a_ev*(pp2 - pp2mid)/pp2denom + b_F*physics::me/(eE + physics::me))
									/(1 + b_F*physics::me/(eE + physics::me));
					
					// Scale the intensity to account for probability
					// Note: Normalization is to N(Decays), not N(DetectedDecays)
					double intensTmp = intens*(P_p2*e2prob)/((double)(this->npos)*(double)(npp))/2;
					
					// cout << "ie: " << ie << ", ide: " << ide << ", ipos: " << ipos << ", ipp: " << ipp << endl;
					g_nFillChan2= 0;
					this->FillChannels(ioffset, -1., -1., A_cosThetaMin+1e-6, this->cosThetaMax, pp2, intensTmp, paramsATmp);
					g_nFillChan= max( g_nFillChan, g_nFillChan2);
					
				}
			}
		}
	}
	
	/* TODO (STEFAN) - ressurect this part later, not needed for simulation	
	if (tailStruct != NULL) {
		// std::cout << "Running tailStruct!" << std::endl;
		// I'm assuming fixed values for smearing the energy here.
		// This'll need to be merged later...
		this->reconstructEe(calibration);
	}
	*/
}

void methodAFit::reconstructEe(double* paramsEe) {
	// TODO make reconstructEe actually work in this context 
	// I'm presently using e2_start and e2_step as fixed values but really
	// I need to make those things a little bit of a smear
	
	// This is a non-continuous function but it's fine (for now)
	double calEe    = paramsEe[0];
	double nonlinEe = paramsEe[1];
	
	// I'm putting these into a separate pointer for now since it's 
	// a little easier to categorize these
	double missDet  = this->tailStruct[0];
	double tailFrac = this->tailStruct[1];
	double tailVal  = this->tailStruct[2];
	
	// Electron energy reconstruction smears the events
	if (missDet > 0) { // Some dumb error conditions
		if (missDet > 1) { missDet = 1; }
		
		for (int ie = 1; ie < this->e2_npts; ie++) {
			// energy bin which gets detected at lower Ee
			int i0 = ie*this->Et2_npts;
			
			//double offsetTmp = this->e2_start - this->e2_step/2;
			double tmpCal    = calEe*this->e2_step;
						
			double eE = eECal((double)ie + 0.5, this->e2_start, tmpCal, nonlinEe);
			// This is different than every other energy extraction? SB: Only because I haven't touched it for a long time
			//double e2 = e2recr(0, (e2_start + (ie + 0.5)*e2_step), calEe, nonlinEe);
			
			double tail   = tailFrac*missDet*eE;//300000;
			double tailXi = tailVal*eE/this->lenSpec; // I'm removing a scale factor of 1000 in tailXi
			
			double eRespC = tail/tailXi/(std::exp((eE/this->lenSpec)/tailXi) - 1); 
						
			for (int ide = 0; ide < ie; ide++) {
				
				// Calibrate the neighboring bins to weight things
				double eBinR = eECal(ide + 1, this->e2_start, tmpCal, nonlinEe);
				double eBinL = eECal(ide,     this->e2_start, tmpCal, nonlinEe);

				// Detector response for electrons grows exponentially
				double intens = eRespC*tailXi*
								(std::exp((eBinR/this->lenSpec)/tailXi)-std::exp((eBinL/this->lenSpec)/tailXi));
								
				// Loop through lower energy parts to correct the array
				for (int ichan = 0; ichan < this->Et2_npts; ichan++) {
					this->chan[ide*this->Et2_npts + ichan] += this->chan[i0 + ichan]*intens;
					this->chan[i0 + ichan]            = this->chan[i0 + ichan]*(1 - intens);
				}
			}
		}
	}	
}

void WriteStatus( double* par) { 
	FILE* file;
	std::string datafilename = "status.dat";
	
	file = fopen(datafilename.c_str(), "w");
	if (file == NULL) {
    	perror( "Cannot open output data file");
		return;
	}
	for (int i= 0; i <= 22; i++) 
		fprintf (file, "%13.9lf", par[i]);
	fclose (file);
}

void ReadStatus( double* par) { 
	FILE* file;
	std::string datafilename = "status.dat";
	
	file = fopen(datafilename.c_str(), "r");
	if (file == NULL) {
    	perror( "Cannot open output data file");
		return;
	}
	for (int i= 0; i <= 22; i++) { 
		int istat= fscanf (file, "%lf", &(par[i]));
		if (istat != 1)
			  cout << "Parameter "<< i << " not read correctly from " << datafilename << "(error "<< istat << ")" << endl;
	}
	fclose (file);
}

// This is the "main" thing that we're using to fit. 
void chi2(int & /*npar*/, double* /*grad*/, double &fval, double* par, int /*iflag*/) {
	// This is a chi^2 function defined in the correct way for MINUIT
	// It goes through our dataset and tries to fit our g_fitObject

	double chi_2 = 0.;
	g_nSimulateET2Spec++; // increase counter for number of function calls
	g_nFillChan= 0;

	// Test if TOF is monotonous and reasonable
	double paramsATmp[6] = { par[3],
							par[4]+5.0-par[9],
							par[5],
							par[6],
							par[7],
							par[8] };
	
	// Tests of parameters reasonable
	if ((par[0] < -1) || (par[0] > 1)) {
			cout << "a_ev = " << par[0] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((par[1] < -1) || (par[1] > 1)) {
			cout << "b_Fierz = " << par[1] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((par[2] < 0) || (par[2] > 10)) {
			cout << "log_10_N = " << par[2] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((paramsATmp[1] < 4.8) || (paramsATmp[1] > 5.5)) {
			cout << "L_zDV = " << paramsATmp[1] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((par[5] < -5) || (par[5] > 5)) {
			cout << "alpha = " << par[5] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((par[6] < -1) || (par[6] > 1)) {
			cout << "beta = " << par[6] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((par[7] < -5) || (par[7] > 5)) {
			cout << "gamma = " << par[7] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	if ((par[8] < -1) || (par[8] > 1)) {
			cout << "eta = " << par[8] << " rejected" << endl;
			fval= 1e10;
			return; // Reject this parameter set.
	}
	// Test if t_p(cos Theta) is monotonous 
	double ntest= 50;
	//double TOFxprevious= g_fitObject->TOF_MethodA(par[3], 0., paramsATmp);
	double TOFxprevious= 100.;
	for (int icos = 1; icos <= ntest; icos++) {
		double cosTheta0= EnsureRange( par[3]+((double)icos)*(1.-par[3])/ntest, par[3], 1.);
		double TOFx= g_fitObject->TOF_MethodA(cosTheta0, 0., paramsATmp);
		if ((TOFx >= TOFxprevious) || (TOFx < 4.5) || (TOFx > 5.8)) {
			//cout << "cosTheta0 = "<< cosTheta0 << ", alpha = " << par[5] << ", beta = " << par[6] << ", gamma = " << par[7] << ", eta= " << par[8]<< " rejected (" << TOFx << ")" << endl;
			fval= 1e10;
			return; // Reject this parameter set. TOF has to fall with cosTheta0.
		}
		TOFxprevious= TOFx;
	}	

	// Compute Chi2
	//cout << "Computing chi2 #" << g_nSimulateET2Spec << ": log_10_N= " << par[2] << ", a_ev = " << par[0] << ", b_Fierz = " << par[1] << ", LNabM5 = " << par[4] << ", alpha = " << par[5] << ", beta = " << par[6] << ", gamma = " << par[7] << ", eta= " << par[8]<< "..." << endl;
	WriteStatus( par);
	
	
	g_fitObject->resetChan();
	g_fitObject->simulateET2SpecMethodA(par);
	//cout << g_nSimulateET2Spec << " !" << endl;

	for (int iE = 1; iE <= fitOpt::numBinsX; iE++) {
		
		double itp_low=0 , itp_up= 0;
		if (FitRegionIncludesEdges == true) {
			itp_low= TMath::FloorNint( g_fitObject->Et2_to_di( h_low_outer->GetBinContent(iE	)));
			itp_up= TMath::FloorNint( g_fitObject->Et2_to_di( h_up_outer->GetBinContent(iE)));
		} else {
			itp_low= TMath::FloorNint( g_fitObject->Et2_to_di( h_low_inner->GetBinContent(iE	)));
			itp_up= TMath::FloorNint( g_fitObject->Et2_to_di( h_up_inner->GetBinContent(iE)));
		}	
		if (itp_up <= itp_low)
			continue;		
		
			/*
			double max = 0.;
			for (int itp = 1; itp <= fitOpt::numBinsY; itp++) { 
				//if (itp < g_fitObject->getEt2_skip()) { continue; }
				//if (iE < g_fitObject->gete2_skip()) { continue; } 
				if (itp < 8) { continue; }
				if (iE < 8) { continue; } 
			
				double output = g_fitObject->getChan((iE-1)*fitOpt::numBinsY + (itp-1));
				if (max < output) { max = output; } 
			}
			*/
		
		if (itp_low < 1)
			itp_low= 1;
		if (itp_up > fitOpt::numBinsY)
			itp_up= fitOpt::numBinsY;
		for (int itp = itp_low; itp <= itp_up; itp++) {
			//if (itp < g_fitObject->getEt2_skip()) { continue; }
			//if (iE < g_fitObject->gete2_skip()) { continue; } 
			// if (itp < 8) { continue; }
			// if (iE < 8) { continue; } 
			
			// Calculate chi2 for each bin
			double output = g_fitObject->getChan((iE-1)*fitOpt::numBinsY + (itp-1));
			//if (output > 0.75*max) { continue; } // If we want to just fit edges
			
			double binContent = h_data->GetBinContent(iE, itp);
			double binError = h_data->GetBinError(iE,itp);
		
			// And now it's just adding things
			// if (binContent == 0 && output == 0) { continue; }
			if (binError != 0) { 
				chi_2 += ((output - binContent)*(output- binContent))/(	binError*binError); 
			    // (binContent != 0) { g_chi_2 += ((output - binContent)*(output- binContent))/binContent; }
			} else {
				chi_2 += ((output-binContent)*(output-binContent));
				//cout << "Something wrong with binerror!" << endl;
			}

			
		}
	}
	
	//catch (...) {
	//	chi_2= 1e10;
	//	cout << "Evaluation crashed! ( log_10_N= " << par[2] << "a_ev = " << par[0] << "b_Fierz = " << par[1] << "LNabM5 = " << par[4] << ", alpha = " << par[5] << ", beta = " << par[6] << ", gamma = " << par[7] << ", eta= " << par[8]<< " rejected (" << TOFxprevious << ")" << endl;
	//}	
	fval = chi_2;

	/*for (int iE = 1; iE <= fitOpt::numBinsX; iE++) {
		
		double max = 0.;
		for (int itp = 1; itp <= fitOpt::numBinsY; itp++) { 
			//if (itp < g_fitObject->getEt2_skip()) { continue; }
			//if (iE < g_fitObject->gete2_skip()) { continue; } 
			if (itp < 8) { continue; }
			if (iE < 8) { continue; } 
			
			double output = g_fitObject->getChan((iE-1)*fitOpt::numBinsY + (itp-1));
			if (max < output) { max = output; } 
		}
		
		for (int itp = 1; itp <= fitOpt::numBinsY; itp++) {
			//if (itp < g_fitObject->getEt2_skip()) { continue; }
			//if (iE < g_fitObject->gete2_skip()) { continue; } 
			if (itp < 8) { continue; }
			if (iE < 8) { continue; } 
			
			// Calculate chi2 for each bin
			double output = g_fitObject->getChan((iE-1)*fitOpt::numBinsY + (itp-1));
			//if ((output < 0.75*max)) { continue; } // If we want to just fit edges
			
			double binContent = h_data->GetBinContent(iE, itp);
			double binError = h_data->GetBinError(iE,itp);
			
			// And now it's just adding things
			if (binContent == 0 && output == 0) { continue; }
			if (binError != 0) { g_chi_2 += ((output - binContent)*(output- binContent))/(binError*binError); }
			//if (binContent != 0) { g_chi_2 += ((output - binContent)*(output- binContent))/binContent; }
			else g_chi_2 += ((output-binContent)*(output-binContent));
			g_ndf++;
		}
	}
  
	fval = g_chi_2;*/
	g_nFillChan++; // increase counter for number of function calls
}

void methodAFit::fillHistsDebug(double* params, int* nfitdata) {
	// This creates a histogram of the fitted actual data
	// as well as some residual plots.
	
	// Mapping to other variables (this does nothing
	/*double a           = params[0];
	double b_Fierz     = params[1];
	double intens      = params[2];
	
	double cosThetaMin = params[3];
	double LNabM5      = params[4];
	double alpha       = params[5];
	double beta        = params[6];
	double gamma       = params[7];
	double eta         = params[8];
	
	double z0_center   = params[9];
	double z0_width    = params[10];
		
	double calEe       = params[11];
	double EeNonLinearity= params[12];
	double sigmaEe       = params[13];*/
	
	(*nfitdata) = 0;
	this->resetChan();
	this->simulateET2SpecMethodA(params);
		
	// Loop through channels and  fill up two histograms for our fit
	for (int iE = 1; iE <= fitOpt::numBinsX; iE++) {
		double itp_low=0 , itp_up= 0;
		if (FitRegionIncludesEdges == true) {
			itp_low= TMath::FloorNint( g_fitObject->Et2_to_di( h_low_outer->GetBinContent(iE)));
			itp_up= TMath::FloorNint( g_fitObject->Et2_to_di( h_up_outer->GetBinContent(iE)));
		} else {
			itp_low= TMath::FloorNint( g_fitObject->Et2_to_di( h_low_inner->GetBinContent(iE)));
			itp_up= TMath::FloorNint( g_fitObject->Et2_to_di( h_up_inner->GetBinContent(iE)));
		}	
		if (itp_up <= itp_low)
			continue;		

		if (itp_low < 1)
			itp_low= 1;
		if (itp_up > fitOpt::numBinsY)
			itp_up= fitOpt::numBinsY;
		for (int itp = itp_low; itp <= itp_up; itp++) {
			//if (itp < this->Et2_skip) { continue; } 
			double binContent = h_data->GetBinContent(iE, itp);
			double output = this->getChan((iE-1)*(fitOpt::numBinsY) + (itp-1));
			//if (!(output == 0) && !(binContent == 0)) { 
				//std::cout << "Bin: " << i << " " << j << std::endl;
				//std::cout << "  Data: " << binContent << std::endl;
				//std::cout << "  Calc: " << output << std::endl;
			//}
			h_fit2D->SetBinContent(iE, itp, output);
			if ((output == 0) || (binContent == 0)) { continue; }
			h_residual->SetBinContent(iE, itp, output - binContent);
			(*nfitdata)++;
		}
	}
	/*for (int iE = 1; iE <= fitOpt::numBinsX; iE++) {
		//if (iE < this->e2_skip) { continue; } 
		if (iE < 8) { continue; } 
		for (int itp = 1; itp <= fitOpt::numBinsY; itp++) {
			//if (itp < this->Et2_skip) { continue; } 
			if (itp < 8) { continue; } 
			double binContent = h_data->GetBinContent(iE, itp);
			double output = this->getChan((iE-1)*(fitOpt::numBinsY) + (itp-1));
			//if (!(output == 0) && !(binContent == 0)) { 
				//std::cout << "Bin: " << i << " " << j << std::endl;
				//std::cout << "  Data: " << binContent << std::endl;
				//std::cout << "  Calc: " << output << std::endl;
			//}
			h_fit2D->SetBinContent(iE, itp, output);
			if ((output == 0) || (binContent == 0)) { continue; }
			h_residual->SetBinContent(iE, itp, output - binContent);
		}
	}*/
}

// This is a root function and so this is the equivalent of "main"
int fit_noE(int argc, char** argv) {
	
	//g_fitObject = new methodAFit();
	//TTime starttime = gSystem->Now();

	// File reading stuff. It's a .csv now (even though it's listed as .dat)
	FILE* file;
	//file = fopen("/home/ffi/Nab/data/simFiles/noE_1B_seed1001-1100_160Ebins_801tp2bins_.dat", "r");
	//file = fopen("/home/ffi/Nab/data/output/teardropBS_3516_2.csv", "r");
	//file = fopen("/home/ffi/Nab/data/output/simTeardrop_noThresh.csv", "r");
	//file = fopen("/home/ffi/Nab/data/output/TeardropBackScatter_1000.csv","r");
	file = fopen("/home/ffi/Nab/data/output/TeardropBackScatter_1000.csv","r");
	//file = fopen(datafilename.c_str(), "r");
	if (file == NULL) {
    	perror( "Cannot open input data file");
		return (-1);
	}

	double totEvents = 0;
	int ndata= 0, nfitdata=0;
	int c = getc(file);
	while (c != EOF) { // loop to read each line of the file and input data into histogram
		int ix, iy = 1;
		double numEvents = 0.;
		
		fscanf(file, "%i %i %lf", &ix, &iy, &numEvents);
		//std::cout << "Loaded: " << std::endl;
		c = getc(file);
		//std::cout<< ix << " " << iy << " " << numEvents << std::endl;
		if (ix > fitOpt::numBinsX) {
			std::cout << "Error! Too many bins (X)!" << std::endl;
			return -1;
		}
		if (iy > fitOpt::numBinsY) {
			std::cout << "Error! Too many bins (Y)!" << std::endl;
			return -1;
		}
		
		h_data->SetBinContent(ix, iy, numEvents);
		h_data->SetBinError(ix, iy, std::sqrt(numEvents+1));
		//if (numEvents > 10) {
		//	h_data->SetBinError(ix, iy, std::sqrt(numEvents));
		//} else {
		//	h_data->SetBinError(ix, iy, 10);
		//}
		ndata++;
		totEvents += numEvents;
	}
	fclose(file);

	std::cout << "Total Events: " << totEvents << " in " << ndata << " bins" << std::endl;
	h_data->Scale(1./totEvents);
	
	// find fit region
	int ix450= TMath::FloorNint( g_fitObject->e2_to_di( 450000.))+1;
	// cout << "Bin " << ix450 <<" @ " << h_data->GetBinCenter(ix450) << " eV" << endl;
	double EventsInHe2 = 0, nEventsInHe2=0;
	for (int iy = 1; iy <= fitOpt::numBinsY; iy++) {
		EventsInHe2+= h_data->GetBinContent(ix450,iy);
	}	
	double Et2_450_low_outer= 0, Et2_450_low_inner= 0, Et2_450_up_outer= 0, Et2_450_up_inner= 0; 
	for (int iy = 1; iy <= fitOpt::numBinsY; iy++) {
		nEventsInHe2+= h_data->GetBinContent(ix450,iy);
		if (nEventsInHe2 < 0.02*EventsInHe2)
			Et2_450_low_outer= g_fitObject->di_to_Et2( iy-.5);	
		if (nEventsInHe2 < 0.13*EventsInHe2)
			Et2_450_low_inner= g_fitObject->di_to_Et2( iy-.5);
		if (nEventsInHe2 < 0.87*EventsInHe2)
			Et2_450_up_inner= g_fitObject->di_to_Et2( iy-.5);
		if (nEventsInHe2 < 0.98*EventsInHe2)
			Et2_450_up_outer= g_fitObject->di_to_Et2( iy-.5);
	}
	
	for (int ix = 1; ix <= fitOpt::numBinsX; ix++) { // move through e2
	    double e2_guess= g_fitObject->di_to_e2 (ix-1);
		if (e2_guess < 100000)
			continue;	// lower e2 cutoff
		h_low_outer->SetBinContent(ix,
			EnsureRange( std::pow(physics::ppmin(e2_guess)/physics::ppmin(450000.),2)*Et2_450_low_outer,
				std::pow(1/40e-6, 2), std::pow(1/10e-6, 2)));
		h_low_inner->SetBinContent(ix,
			EnsureRange( std::pow(physics::ppmin(e2_guess)/physics::ppmin(450000.),2)*Et2_450_low_inner,
				std::pow(1/40e-6, 2), std::pow(1/10e-6, 2)));
		h_up_outer->SetBinContent(ix,
			EnsureRange( std::pow(physics::ppmax(e2_guess)/physics::ppmax(450000.),2)*Et2_450_up_outer,
				std::pow(1/40e-6, 2), std::pow(1/10e-6, 2)));
		h_up_inner->SetBinContent(ix,
			EnsureRange( std::pow(physics::ppmax(e2_guess)/physics::ppmax(450000.),2)*Et2_450_up_inner,
				std::pow(1/40e-6, 2), std::pow(1/10e-6, 2)));
	}	
	
	/*for (int ix = 1; ix <= fitOpt::numBinsX; ix++) {
	 	for (int iy = 1; iy <= fitOpt::numBinsY; iy++) {
			dataList[(ix-1)*fitOpt::numBinsY + (iy-1)] = h_data->GetBinContent(ix,iy);
			//h_data->SetBinContent(ix, iy, h_data->GetBinContent(ix, iy)/totEvents);
			//h_data->SetBinError(ix, iy, h_data->GetBinError(ix,iy)/totEvents);
		}
	}	
	std::cout << "histogram filled with data" << std::endl;*/
	g_nFillChan= 0; // Reset counter for number of function calls
	TTime starttime = gSystem->Now(); // Start clock that measures run time
    g_nSimulateET2Spec= 0; // Reset counter for number of function calls
	
	TVirtualFitter* minuit = TVirtualFitter::Fitter(0, 23);
	std::cout << "Setting Parameters" << std::endl;
	
	double params[23];
	params[0]= -0.1; // a_ev
	params[1]= 0.0; // b_Fierz
	params[2]= 0.91; // log_10_N
	params[3]= 0.76; // costhetamin
	params[4]= 0.09; // LNabM5
    params[5]= 0.6; // alpha	
	params[6]= -0.05; // beta
	params[7]= 2.1; // gamma
	params[8]= 0.056; // eta
	params[9]= -0.132; // z0_center
	params[10]= 0.08; // z0_width
	params[11]= 0.01; // missdet
	params[12]= 0; // tailfrac
	params[13]= 0.13; // tailVal
	params[14]= 0; // hvMapMin1
	params[15]= 0; // hvMap0
	params[16]= 0; // hvMap1
	params[17]= 0; // hvMap2
	params[18]= 0; // hvMap3
	params[19]= 0; // hvMap4
	
	// Fixed HV parameter hardcoded in from 
	// https://nabcms.phys.virginia.edu/issues/130#note-16
	params[14]= -0.0217; // hvMapMin1
	params[15]= 0; // hvMap0
	params[16]= 0.507; // hvMap1
	params[17]= -0.960; // hvMap2
	params[18]= 0.105; // hvMap3
	params[19]= -0.0457; // hvMap4
	
	params[20]= 1.; // calEe
	params[21]= 0; // EeNonLinearity
	params[22]= 1.5; // sigmaEe_keV

	//ReadStatus( params);
	
	// Lol let's go ahead and put 20 parameters in the fit function.
	// We'll have to import some of these from data to actually solve
	// To actually do a fit, we need to do this in two steps.
	// First, we fit the detector response function (parameters 4-8 + 10)
	// Then, we fix those and fit a_ev, intens, and b_Fierz.
	//minuit->SetParameter(0, "a_ev", -0.103145252, 0.00012800125, 0, 0);
	//minuit->SetParameter(0, "a_ev", -0.102918, 0.000109747, 0, 0);
	minuit->SetParameter(0, "a_ev", params[0], 0.01, -1, 1);
	//minuit->FixParameter(0);//
	minuit->SetParameter(1, "b_Fierz", params[1], 0.01, -0.5, 0.5);
	//minuit->SetParameter(1, "b_Fierz", 0.0, 3.13200e-4, 0, 0);
	//minuit->FixParameter(1);
	minuit->SetParameter(2, "intens", params[2], 0.05, 0, 0);
	//minuit->SetParameter(2, "intens", 9.8159438 /*6545514700*/ /*9.81827205*/, 0.01, 0, 0);
	//minuit->SetParameter(2, "intens", 0.3 /*6545514700*/ /*9.81827205*/, 0.1, 0, 0);
	//minuit->FixParameter(2);
	minuit->SetParameter(3, "costhetamin", params[3], 0.01, 0, 0);
	minuit->FixParameter(3);
	minuit->SetParameter(4, "LNabM5", params[4], 0.1, -0.2, 0.7);
	//minuit->SetParameter(4, "LNabM5", 0.22088e-01,8.26382e-05,0,0);
	//minuit->SetParameter(4, "LNabM5", 0.2239050,0.001,0,0);//*0.136506031*/, 0.000368683024, 0, 0);
	//minuit->SetParameter(4, "LNabM5", 0.224482/*0.136506031*/, 0.000368683024, 0, 0);
	//minuit->SetParameter(4, "LNabM5", 0.2, 0.1, 0, 0);
	//minuit->FixParameter(4);//
	minuit->SetParameter(5, "alpha", params[5],0.3,0.,0.);//, 0.0146502327, 0, 0);
	//minuit->SetParameter(5, "alpha", 0.76461203, 0.0146502327, 0, 0);
	//minuit->SetParameter(5, "alpha", 0.75, 0.1, 0, 0);
	//minuit->FixParameter(5);//
	minuit->SetParameter(6, "beta", params[6],0.3,0,0);
	//minuit->SetParameter(6, "beta", 0.331289/*-0.532039282*/, 0.00355904/*0.127939658*/, 0, 0);
	//minuit->SetParameter(6, "beta", 0.3, 0.1, 0, 0);
	//minuit->FixParameter(6);
	minuit->SetParameter(7, "gamma", params[7],0.5,0.,0.);
	//minuit->SetParameter(7, "gamma", 4.5631/*2.10570387*/, 0.0216455/*0.422229823*/, 0, 0);
	//minuit->SetParameter(7, "gamma", 4, 0.1, 0, 0);
	minuit->FixParameter(7);//
	minuit->SetParameter(8, "eta", params[8],0.01,0.,0.);
	//minuit->SetParameter(8, "eta", 0.1, 0.1, 0, 0);
	//minuit->FixParameter(8);//
	minuit->SetParameter(9, "z0_center", params[9], 0.01, 0, 0);
	minuit->FixParameter(9);
	minuit->SetParameter(10, "z0_width", params[10],0.01,0.,0.);
	minuit->FixParameter(10);
	//minuit->SetParameter(10, "z0_width", 0.0807151375, 0.00140325194, 0, 0);
	//minuit->SetParameter(10, "z0_width", 0.1, 0.1, 0, 0);
	
	//minuit->SetParameter(11, "missDet", 0.01, 0.01, 0, 0);
	minuit->SetParameter(11, "missDet", params[11], 0.01, 0, 0);
	minuit->FixParameter(11);
	//minuit->SetParameter(12, "tailFrac", 0.7, 0.01, 0, 0);
	minuit->SetParameter(12, "tailFrac", params[12], 0.01, 0, 0);
	minuit->FixParameter(12);
	//minuit->SetParameter(13, "tailVal", 0.13, 0.01, 0, 0);
	minuit->SetParameter(13, "tailVal", params[13], 0.01, 0, 0);
	minuit->FixParameter(13);
	//minuit->SetParameter(14, "hvMapMin1", -1, 0.01, 0, 0);
	minuit->SetParameter(14, "hvMapMin1", params[14], 0.01, 0, 0);
	minuit->FixParameter(14);
	//minuit->SetParameter(15, "hvMap0", 1, 0.01, 0, 0);
	minuit->SetParameter(15, "hvMap0", params[15], 0.01, 0, 0);
	minuit->FixParameter(15);
	//minuit->SetParameter(16, "hvMap1", 1, 0.01, 0, 0);
	minuit->SetParameter(16, "hvMap1", params[16], 0.01, 0, 0);
	minuit->FixParameter(16);
	//minuit->SetParameter(17, "hvMap2", 1, 0.01, 0, 0);
	minuit->SetParameter(17, "hvMap2", params[17], 0.01, 0, 0);
	minuit->FixParameter(17);
	//minuit->SetParameter(18, "hvMap3", 1, 0.01, 0, 0);
	minuit->SetParameter(18, "hvMap3", params[18], 0.01, 0, 0);
	minuit->FixParameter(18);
	//minuit->SetParameter(19, "hvMap4", 2, 0.01, 0, 0);
	minuit->SetParameter(19, "hvMap4", params[19], 0.01, 0, 0);
	minuit->FixParameter(19);
	minuit->SetParameter(20, "calEe", params[20], 0.01, 0, 0);
	minuit->FixParameter(20);
	minuit->SetParameter(21, "EeNonLinearity", params[21], 0.01, 0, 0);
	minuit->FixParameter(21);
	minuit->SetParameter(22, "sigmaEe_keV", params[22], 0.1, 0, 0);
	minuit->FixParameter(22);
	
	std::cout << "Parameters Set" << std::endl;
	//void chi2(int & nPar, double* grad, double &fval, double *par, int iflag);
	minuit->SetFCN(chi2);

	std::cout << "Setting Function" << std::endl;
	double arglist[100];
	arglist[0] = 0;
	minuit->ExecuteCommand("SET PRINT", arglist, 1);

	arglist[0] = 1; // error level for chi2 minimization
	minuit->ExecuteCommand("SET ERR", arglist, 1);

	arglist[0] = 5000; // num calls (5000)
	arglist[1] = 0.01; // tolerance (0.01)

	// This is actually the "FIT" function that we want to call.
	//minuit->ExecuteCommand("MINIMIZE", arglist, 2);
	//std::cout << "MIGRAD finished!\n" << std::endl;
	//minuit->ExecuteCommand("MINOS",  arglist, 2);
	
	TTime endtime = gSystem->Now(); // End clock that measures run time
    TTime duration= endtime-starttime;
	
	// Fill a histogram of our optimal values
	cout << "Final assessment of fit results:" << endl;
	double minParams[23];
	double parErrors[23];
	for (int i = 0; i < 23; i++) {
		minParams[i] = minuit->GetParameter(i);
		parErrors[i] = minuit->GetParError(i);

		std::cout << minuit->GetParName(i) << ":\t\t" << minParams[i] << "\t\terr:\t" << parErrors[i] << std::endl;
	}
	
	/*std::cout << "Printing Covariance Matrix: " << std::endl;
	for (int i = 0; i < 23; i++) {
		
		for (int j=0; j < 23; j++) {
			std::cout << "\t";
			std::cout << minuit->GetCovarianceMatrixElement(i,j);
			
		}
			std::cout << "\n\n";
			
	}*/

	g_fitObject->fillHistsDebug(minParams, &nfitdata);

	// Pulling stats from Minuit's internal pointers
	double chi_2, edm, errdef;
	int nvpar, nparx;
	minuit->GetStats(chi_2, edm, errdef, nvpar, nparx);
	
	std::cout << "\nchi2:\t\t" << chi_2 << std::endl;
	std::cout << "Data points: " << ndata << ", in fit region: " << nfitdata
				<< ", NDF:" << nfitdata-nvpar << std::endl;
	cout << "Number of function calls:\t" << g_nSimulateET2Spec++  << endl;
	std::cout << "Duration of fitting routine:\t" << (duration).AsString() << " ms" << std::endl;
	#ifndef __CLING__
	TApplication plotApp("pApp",&argc,argv);
	#endif
	// Now we plot
	TCanvas *c1 = new TCanvas("c1", "MethodAFitOutput", 1200, 1200);
	c1->Divide(2,2);
	c1->cd(1);
	h_data->Draw("COLZ");
	c1->cd(2);
	h_fit2D->Draw("COLZ");
	c1->cd(3);
	h_residual->Draw("COLZ");
	c1->cd(4);
	h_residual->Draw("LEGO");
	gStyle->SetOptStat(0);
	gStyle->SetOptFit(0);
	gStyle->SetPadLeftMargin(0.15);
	gStyle->SetPadRightMargin(0.15);
	gStyle->SetTitleOffset(1.5);

	// This adds potential fit boundaries
	c1->cd(1);
	h_low_outer->Draw("same");
	h_up_outer->Draw("same");
	h_low_inner->Draw("same");
	h_up_inner->Draw("same");
	c1->Update();
	#ifndef __CLING__
	std::cout << "Use CTRL+C to exit!" << std::endl;
	plotApp.Run();
	#endif
	//std::cout << std::endl << gSystem->Now().operator-=(starttime).AsString() << " ms" << std::endl;

	return (0);
}

extern "C" {
	methodAFit* methodAFit_new(){ return new methodAFit(); } 
	void methodASpec(methodAFit* methodA, double* params) { 
		methodA->resetChan();
		return methodA->simulateET2SpecMethodA(params); 
	}
}

#endif
