#include <iostream>
#include <cmath>
#include <random>
#include <TFile.h>
#include <TTree.h>

// Distance from the lower TOF plane to the RICH is 8.65 cm.

// "Event" refers to the struct, while "event" refers to each generated event,
// together with its physical characteristics and measurements.

// x_0: the true physical value of x, which is never known exactly by the detector.
// x_subsystem: the value of x measured by the corresponding detector subsystem.

// -1 is used as a non-physical flag for events that were not measured or detected.


const double u_to_GeV = 0.93149410242;   // Conversion from atomic mass units to GeV.
 
 
struct Event{   // Definition of the Event struct. Each event is characterized by the isotope
                // to which it corresponds, the true physical quantities of the particle,
                // and the measurements of those quantities by each detector subsystem.

    int Z;          
    int A;          

    double m_0;     // Mass in GeV.
    double m_TOF;
    double m_RICH;             

    double P_0;   
    double P;

    double theta;   
    double phi;     

    double x;       // These x and y variables correspond to the upper TOF plane;
    double y;       // this is the first interaction point of the particle with the detector.

    double beta_0;   
    double beta_TOF;       
    double beta_RICH;
    
    double T_0;     // T = m(gamma - 1).
    double T_TOF;
    double T_RICH;

    double omega;   // Statistical weight.

    bool trigger;   // Boolean variable indicating whether or not the particle was detected.
                    // The TOF trigger is expected to be true for all generated events
                    // in this version of the program.
};





class AMS02Simulator{

private:

    double P_min;
    double P_max;

    int Z;


    // TOF geometry.

    double TOF_side;
    double TOF_half_side;
    double TOF_z_dist;
    double TOF_z_dist_vn;    // In natural units.
    double TOF_sigma_t;              


    // RICH resolution.

    double RICH_naf_A;
    double RICH_naf_B;
    double RICH_agl_A;
    double RICH_agl_B;
    double RICH_naf_E0;
    double RICH_agl_E0;
    double RICH_naf_side;
    double RICH_agl_radius;
    
    
    double TOF_RICH_dist;
    
    // Tracker resolution.
    
    double tracker_r;       // Temporary.


    double P_mean;


    // C++ random number generators.

    std::mt19937 generator;                         // Pseudo-random number generator.
    std::uniform_real_distribution<double> uniform; // Uniformly distributed random numbers
                                                    // over a specified range [a, b).
    std::normal_distribution<double> gaussian;      // Gaussian distribution N(0,1), with
                                                    // mean 0 and standard deviation 1.



public:


    AMS02Simulator(double p_min, double p_max):
    P_min(p_min),
    P_max(p_max),
    generator(std::random_device{}()),              
    uniform(0.0,1.0),                                                                
    gaussian(0.0,1.0)
    
    {

        TOF_side = 130.0;

        TOF_half_side = TOF_side/2.0;

        TOF_z_dist = 127.3;
        
        TOF_z_dist_vn = 127.3 * 3.33564e-9;       // Conversion to natural units.
        
        TOF_sigma_t = 160e-12;


        RICH_naf_A = 12.73e-4;

        RICH_naf_B = 3.76e-4;
        
        RICH_agl_A = 3.57e-4;
        
        RICH_agl_B = 0.61e-4;

        RICH_naf_E0 = 0.938;
        
        RICH_agl_E0 = 2.983;
        
        RICH_naf_side = 34.5;
        
        RICH_agl_radius = 57.0;
        
        TOF_RICH_dist = 40.0;
        
        
        tracker_r = 0.01;                        // Temporary.



        P_mean =(P_max-P_min)/log(P_max/P_min);

    }


    Event generate_event();         

    void generate_isotope(Event &event);

    double generate_rigidity();

    void generate_cinematica(Event &event);

    double calculate_beta_0(double P_0,double m_0,int Z);

    void calculate_peso(Event &event);

    bool triggerTOF(Event &event);
    
    double sigma_beta_RICH(Event &event);

    double sigma_beta_TOF(Event &event);

    double sigma_P(Event &event);


};





void AMS02Simulator::generate_isotope(Event &event){

    double fracao_B10 = 0.25;       // Approximate relative abundance of the B-10 isotope
                                    // in cosmic rays.

    double u = uniform(generator);


    if(u < fracao_B10)
    {
        event.A=10;  
    }

    else
    {
        event.A=11;  
    }

    event.m_0=event.A*u_to_GeV;    
    
    event.Z=5;                     // The charge is constant for the isotopes considered
                                    // in this study.


}





Event AMS02Simulator::generate_event(){  // Function that connects all components of the simulation
                                         // and fills the Event struct. In other words, this is
                                         // the function that actually generates an event.

    Event event;
    
    generate_isotope(event);        

    event.P_0 = generate_rigidity();   

    event.beta_0 = calculate_beta_0(event.P_0,event.m_0,event.Z); 
    
    event.T_0 = event.m_0*(1/sqrt(1 - event.beta_0*event.beta_0) - 1);
    
    generate_cinematica(event);     

    event.P = event.P_0 + sigma_P(event) * gaussian(generator);  // A temporary tracker resolution
                                                                  // is used.

    event.trigger = triggerTOF(event);
    
    if(event.trigger == true){
        
        event.beta_TOF = 1/(1/event.beta_0 + sigma_beta_TOF(event)/(TOF_z_dist_vn/cos(event.theta)));
        // If trigger == true, the TOF measures the particle's beta.
    
        if(sigma_beta_RICH(event) != -1){                                
    
            event.beta_RICH = event.beta_0 + sigma_beta_RICH(event)*gaussian(generator);

        }
    
        else{

            event.beta_RICH = -1;       
        
        }
    
    }
    
    else{
        
        event.beta_RICH = -1;   // If trigger == false, both subsystems fail to measure
                                // the particle's velocity.
        event.beta_TOF = -1;
    
    }

    if(event.beta_TOF != -1){
        event.m_TOF = event.Z*event.P/((1.0/ sqrt(1.0-event.beta_TOF*event.beta_TOF))*event.beta_TOF);   
    }
    
    else{
        event.m_TOF = -1;
    }
        
    if(event.beta_RICH != -1){
        event.m_RICH = event.Z*event.P/((1.0/ sqrt(1.0-event.beta_RICH*event.beta_RICH))*event.beta_RICH);
    }
    
    else{
        event.m_RICH = -1;
    }
    
    if(event.m_TOF != -1){
        event.T_TOF = event.m_TOF*(1/sqrt(1 - event.beta_TOF*event.beta_TOF) - 1);
        // T = m(gamma - 1).
    
    }
    
    else{
        event.T_TOF = -1;
    
    }
        
    if(event.m_RICH == -1){
        event.T_RICH = -1;
        
    }

    else{
    
        event.T_RICH = event.m_RICH*(1/sqrt(1 - event.beta_RICH*event.beta_RICH) - 1);
        // T = m(gamma - 1).
        
    }

    calculate_peso(event);        


    return event;

}





double AMS02Simulator::generate_rigidity(){

    double u = uniform(generator);


    double logP = log(P_min) + u*(log(P_max)-log(P_min));


    return exp(logP);

}





double AMS02Simulator::calculate_beta_0(double P_0,double m_0,int Z){

    double p = Z*P_0;


    double E = sqrt(p*p + m_0*m_0);


    return p/E;

}





void AMS02Simulator::generate_cinematica(Event &event){
    
    // Generate random variables between 0 and 1.

    double u_1 = uniform(generator);           
    double u_2 = uniform(generator);           
    double u_3 = uniform(generator);
    double u_4 = uniform(generator);


    event.phi = 2*M_PI*u_2;


    event.x = -TOF_half_side + u_3*TOF_side;


    event.y = -TOF_half_side + u_4*TOF_side;


    double cos_phi = cos(event.phi);
    double sin_phi = sin(event.phi);

   
    double dx_max;
    double dy_max;

    if (fabs(cos_phi) > 1e-12)
        dx_max = (TOF_half_side - fabs(event.x)) / fabs(cos_phi);
    else
        dx_max = 1e30;

    if (fabs(sin_phi) > 1e-12)
        dy_max = (TOF_half_side - fabs(event.y)) / fabs(sin_phi);
    else
        dy_max = 1e30;

    double d_max = std::min(dx_max, dy_max);

    // d = TOF_z_dist * tan(theta).
    double theta_max = atan(d_max / TOF_z_dist);

    // Generate only values of theta such that trigger == true.
    event.theta = acos(1.0 - u_1 * (1.0 - cos(theta_max)));
}





double J(double P_0){

    return std::pow(P_0,-2.7);      // First approximation.

}




void AMS02Simulator::calculate_peso(Event &event){

    event.omega = (event.P_0 * J(event.P_0))/(P_mean * J(P_mean));

}





double AMS02Simulator::sigma_P(Event &event){

    return tracker_r * event.P_0;   // Temporary.

}





double AMS02Simulator::sigma_beta_RICH(Event &event){ 


    double p = event.Z*event.P_0;

    double E = sqrt(p*p + event.m_0*event.m_0);

    double E_kn = (E-event.m_0)/event.A;
    
    double resolution = 0.0;
    
    double cos_theta = cos(event.theta);

    double sin_theta = sin(event.theta);
    
    double tan_theta = sin_theta/cos_theta;
    
    double x_RICH = event.x + 8.65*tan_theta*cos(event.phi);
    // The variables stored in the struct correspond to the coordinates at the top of
    // the TOF. At the RICH, these coordinates change depending on the trajectory angle.
    
    double y_RICH = event.y + 8.65*tan_theta*sin(event.phi);
    // The distance between the TOF planes and the RICH must therefore be taken into account.
    
    double r = sqrt(x_RICH*x_RICH + y_RICH*y_RICH);
    // Length of the position vector of the particle on the RICH plane.
    
    bool bnaf = abs(x_RICH) < RICH_naf_side/2 && abs(y_RICH) < RICH_naf_side/2;
    
    bool bagl = (r < RICH_agl_radius) && !bnaf;
    
    if(event.beta_0>=0.750 && bnaf == true) {                                                      
    
        resolution = RICH_naf_A - RICH_naf_B*exp(1.0-E_kn/RICH_naf_E0);      
    
    }

    if(event.beta_0 >= 0.953 && bagl == true){
     
        resolution = RICH_agl_A - RICH_agl_B*exp(1.0-E_kn/RICH_agl_E0);
    
    }
    
    if(event.beta_0<0.750 || r > RICH_agl_radius){
        
        return -1;
        
    }

    return event.beta_0*resolution;

}





double AMS02Simulator::sigma_beta_TOF(Event &event){


    return TOF_sigma_t*gaussian(generator);      
    
}





bool AMS02Simulator::triggerTOF(Event &event){

    // The generated theta values are restricted such that the TOF trigger
    // should always be true, apart from possible numerical precision effects.

    double cos_theta = cos(event.theta);

    double sin_theta = sin(event.theta);



    if(cos_theta == 0)

        return false;



    double tan_theta = sin_theta/cos_theta;


    double x_lower = event.x + TOF_z_dist*tan_theta*cos(event.phi);


    double y_lower = event.y + TOF_z_dist*tan_theta*sin(event.phi);


    return (abs(x_lower)<=TOF_half_side && abs(y_lower)<=TOF_half_side);

}




void writeTree(AMS02Simulator &sim, int N){
    
    TFile file("AMS02.root","RECREATE");

    TTree tree("AMS","AMS02 Simulation");
    
    Event event;
     
    tree.Branch("Z",&event.Z,"Z/I");
    tree.Branch("A",&event.A,"A/I");

    tree.Branch("P_0",&event.P_0,"P0/D");
    tree.Branch("beta_0",&event.beta_0,"beta0/D");
    tree.Branch("m_0",&event.m_0,"m0/D");
    tree.Branch("T_0",&event.T_0,"T0/D");

    tree.Branch("P",&event.P,"P/D");
    
    tree.Branch("beta_TOF",&event.beta_TOF,"betaTOF/D");
    tree.Branch("beta_RICH",&event.beta_RICH,"betaRICH/D");

    tree.Branch("m_TOF",&event.m_TOF,"mTOF/D");
    tree.Branch("m_RICH",&event.m_RICH,"mRICH/D");
    
    tree.Branch("T_TOF",&event.T_TOF,"TTOF/D");
    tree.Branch("T_RICH",&event.T_RICH,"TRICH/D");

    tree.Branch("omega",&event.omega,"omega/D");

    tree.Branch("theta",&event.theta,"theta/D");
    tree.Branch("phi",&event.phi,"phi/D");

    tree.Branch("x",&event.x,"x/D");
    tree.Branch("y",&event.y,"y/D");

    tree.Branch("trigger",&event.trigger,"trigger/O");
    
    for(int i=0;i<N;i++)
    {
        event = sim.generate_event();

        tree.Fill();
    }

    tree.Write();

    file.Close();
}





int main(){

    AMS02Simulator sim(1.0,100000.0);   // Object "sim" represents the part of the simulator
                                        // configured to generate events.
    
    writeTree(sim,100000);


    int n_TOF10=0;
    int n_TOF11=0;
    
    int n_RICH10=0;
    int n_RICH11=0;
    

    for(int i=0;i<100000;i++){


        Event event = sim.generate_event();   // For each 'i', one complete event is generated.

        if(event.m_TOF > 0){
            
            double A_TOF=event.m_TOF / u_to_GeV;
            
            if(std::round(A_TOF)==10){
            
                n_TOF10++;
            }

            else
            {
                n_TOF11++;
            }
        }
            
        if(event.m_RICH > 0){
        
            double A_RICH=event.m_RICH / u_to_GeV;
        
            if(std::round(A_RICH)==10){
        
                n_RICH10++;
                
            }

            else{
            
                n_RICH11++;
            }

        }

    }


    std::cout<<"B10_TOF = "<<n_TOF10<<std::endl;

    std::cout<<"B11_TOF = "<<n_TOF11<<std::endl;
    
    std::cout<<"B10_RICH = "<<n_RICH10<<std::endl;

    std::cout<<"B11_RICH = "<<n_RICH11<<std::endl;
    
    if(n_RICH10 + n_RICH11 <= n_TOF10 + n_TOF11){
        
    
        std::cout<<"Number of detected particles = "<<n_TOF10 + n_TOF11<<std::endl;
    
    }
    
    else{
        
        std::cout<<"Error in the code";          // Possible error flag.
        // Every particle detected by the RICH must necessarily also be detected by the TOF.
        // A different result indicates an error.
        
    }
    
    
    return 0;

}