// https://lup.lub.lu.se/luur/download?func=downloadFile&recordOId=8889583&fileOId=8890947
// this is not been test

using namespace std;
using namespace MPI;

template <class matrixclass> class GMRES{
    /**
    * Takes type of matrix needs to be compatible with regular multiplication
    * with mpi_doubleVector
    *
    */
    public:
        // stoping residual, Matrix, rhs,dimesion of the system, approximation, restartsize
        int solver(double stopping_res, matrixclass *Matrix, mpi_doubleVector *rhs,int n, mpi_doubleVector *approx, int restartsize){
            const int max_starts=1e2;
            //Get the local segment length
            const int id =COMM_WORLD.Get_rank();
            //The varibles for this method
            int i, j, k=0, start, ready = 0;
            mpi_doubleVector v(n);
            mpi_doubleVector u(n);
            doubleMatrix R(restartsize,restartsize);
            mpi_doubleMatrix H(restartsize,n);
            doubleVector cosinus(restartsize+1);
            doubleVector sinus(restartsize+1);
            doubleVector w(restartsize+1);
            double currenterror;
            //Create a serial linear solver instance
            serialsolver <doubleVector,doubleMatrix> Solver;
            //Do the first multiplication
            v=(*Matrix)*(*approx);
            v = (*rhs)- v;
            double tmp=sqrt(v*v);
            //store the current error
            currenterror=tmp;
            //Generate and apply the first Housholder transformation
            generateHousholder(v,u,0,&w[0]);
            //set all other indicies to 0
            for(int i=0;i<restartsize+1;i++){
                if(i>0){
                    w[i]=0;
                }
            }
            //Stop if approximation is good from the start
            if(tmp<stopping_res){
                ready=1;
            }
            //-------------------------------------------------
            // Number of starts
            //-------------------------------------------------
            for(start = 0; start <= max_starts && !ready; start++){
                //Do the same as above to start the method
                if(start){
                    v=(*Matrix)*(*approx);
                    v = (*rhs)- v;
                    generateHousholder(v,u,0,&w[0]);
                    for(int i=0;i<restartsize+1;i++){
                        if(i>0){
                            w[i]=0;
                        }
                    }
                }
                //------------------------------------
                // The GMRES Method
                //------------------------------------
                for(k = 0; k < restartsize; k++){
                    //store the current trasformation in H
                    H[k]=u;
                    //make v an unit vector
                    for(j=v.li();j<v.ui();j++){
                        v[j]=0;
                    }
                    v[k]=1;
                    //Apply the last k + 1 Householder transformations in reverse order:
                    for(i = k; i >= 0; i--){
                        applyHouseholder(H[i],v,v,i);
                    }
                    u=(*Matrix)*v;
                    v=u;
                    //Apply last k + 1 Householder transformations:
                    for(i = 0; i <= k; i++){
                        applyHouseholder(H[i],v,v,i);
                    }
                    //Generate and apply the last transformation
                    if(k < n - 1){
                    //Let u be the zero vector
                    for(i=u.li();i<u.ui();i++){
                        u[i]=0;
                    }
                    generateHousholder(v,u,k+1,&tmp);
                    /* Apply this transformation: */
                    for(int i=v.li();i<v.ui();i++){
                        if(i==k){
                            v[k+1]=tmp;
                        }
                        if(i>k+1){
                            v[i]=0;
                        }
                    }
                }
                //A double that has the w[k+1]
                double tmp=0;
                //Generate and apply the givens rotations on v and w
                if(id==0){
                    givens_rotations(v,w,R,sinus,cosinus,k);
                    tmp=w[k+1];
                }
                //Broadcast the current error
                MPI_Bcast(&tmp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                //store the current error
                currenterror=fabs(tmp);
                //Check if the solution is good enough
                if(fabs(tmp) < stopping_res){
                    ready = 1;
                    break;
                }
            }
            //------------------------------------
            // The Solver
            //------------------------------------
            if(k==restartsize){
                k--;
            }
            //Solve the triangular system and transfer it to u
            if(id==0){
                doubleVector x=Solver.solveUpperTriangular(R,w,k+1);
                //transfer the solution to u
                for(i=0;i<k+1;i++){
                    u[i]=x[i];
                }
            }
            //Calculate the new approximation
            for(i = 0; i <= k; i++){
                //Unit vector
                for(j =v.li();j<v.ui();j++){
                    v[j]=0;
                }
                v[i]=1;
                //Apply last i + 1 householder transformations in reverse order:
                for(j = i; j >= 0; j--){
                    applyHouseholder(H[j],v,v,j);
                }
                //Get the coeficiant we wish to multiply the vectors with
                double c=0;
                c=u[i];
                //make sure all parts has it
                MPI_Bcast(&c, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
                v *= c; //multiply with scalar
                (*approx) += v; //Get the new approximation
            }
            //If the error is small enough stop, else continue
            if(currenterror < stopping_res){
                ready = 1;
            }
        }
        //Check if we have done maximum number of starts
        if(start > max_starts){
            start = max_starts;
        }
        return (start * restartsize + k + 1);
    }
    private:
        void givens_rotations(mpi_doubleVector v,doubleVector w,doubleMatrix R,doubleVector sinus,doubleVector cosinus,int k);
        void generateHousholder(mpi_doubleVector x, mpi_doubleVector out,int k,double *alpha);
        void applyHouseholder(mpi_doubleVector t,mpi_doubleVector x,mpi_doubleVector y,int k);
        //Simple sign function
        int sign(double v);
};