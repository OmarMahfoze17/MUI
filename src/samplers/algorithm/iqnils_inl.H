/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2019 Y. H. Tang, S. Kudo, X. Bian, Z. Li, G. E. Karniadakis  *
*                                                                            *
* This software is jointly licensed under the Apache License, Version 2.0    *
* and the GNU General Public License version 3, you may use it according     *
* to either.                                                                 *
*                                                                            *
* ** Apache License, version 2.0 **                                          *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this file except in compliance with the License.           *
* You may obtain a copy of the License at                                    *
*                                                                            *
* http://www.apache.org/licenses/LICENSE-2.0                                 *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
*                                                                            *
* ** GNU General Public License, version 3 **                                *
*                                                                            *
* This program is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by       *
* the Free Software Foundation, either version 3 of the License, or          *
* (at your option) any later version.                                        *
*                                                                            *
* This program is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
* GNU General Public License for more details.                               *
*                                                                            *
* You should have received a copy of the GNU General Public License          *
* along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
*****************************************************************************/

/**
 * @file iqnils_inl.H
 * @author W. Liu
 * @date 24 February 2021
 * @brief Interface Quasi-Newton with Inverse Jacobian from Least Squares model
 * (IQN-ILS) Coupling Method (muiCouplingIQNILS class) of FSI Coupling utility
 */

#ifndef MUICOUPLINGIQNILS_HPP
#define MUICOUPLINGIQNILS_HPP

#include "mui.h"

#include <string>
#include <vector>
#include <Eigen/Dense>
#include <map>
#include <math.h>
#include <algorithm>
#include <numeric>
#include <assert.h>
#include <iterator>
#include <iostream>
#include <stdlib.h>
#include <mpi.h>

#define EIGEN_MPL2_ONLY

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
namespace muiCoupling
{

/*---------------------------------------------------------------------------*\
                 Class muiCouplingIQNILS Declaration
\*---------------------------------------------------------------------------*/

    class muiCouplingIQNILS
    {

    private:

        // Private data

            //-MPI Common World
            MPI_Comm world_;

            //-Vector of MPI local ranks
            std::vector<int> localRankVec_;

            //-MPI Common Rank in global mpi_comm_word
            int CppTotalRank_;

            //-MPI Common Size in global mpi_comm_word
            int CppTotalSize_;

            //-MPI Common Size after split
            int sizeAfterSplit_;

            //-MPI Common Rank
            int CppRank_;

            //-MPI Common Size
            int CppSize_;

            //- Under relaxation factor of the coupling at present iteration
            double undRelxCpl_;

            //- Maximum under relaxation factor of the coupling
            double undRelxCplMax_;

            //- Under relaxation factor of the coupling at previous iteration
            mutable double undRelxCplPrev_;

            const double initundRelxCpl_;

            int pointSize_ = -999;

            int aitkenIterationN_;

            std::vector<double> residualCoupling_;

            std::vector<double> residualDelta_;

            std::vector<double> residualCouplingPrev_;

            std::vector<double> fetchDispCoupling_;

            std::vector<double> deltaDisp_;

            std::vector<double> residualMagSq_;

            double residualMagSqSum_;

            double residualL2Norm_;

            double residualL2NormMax_;

            mutable std::map <int, int> pointMap_;

            std::map <int, int> :: iterator it_;

            Eigen::MatrixXd matrixVk;

            Eigen::MatrixXd matrixWk;

            Eigen::MatrixXd matrixVref;

            Eigen::MatrixXd matrixWref;

            Eigen::MatrixXd matrixR;

            Eigen::MatrixXd matrixQ;

            Eigen::VectorXd vectorAlpha;
            
            bool needCalLocalResidualSize;
            
            int globalResidualCouplingSizeSum_;

            std::vector<int> globalResidualCouplingSize_;

            std::vector<double> globalResidualCoupling_;

            bool globalAlpha;

        // Private Member Functions

          // Math tools
            //- Calculation of point ID
            inline int calcpointN(int pointV)
            {
                it_ = pointMap_.find(pointV);
                if(it_ != pointMap_.end())
                {
                    return (( *it_ ).second);
                }
                else
                {
                    pointMap_.insert(std::pair<int, int>(pointV, (pointMap_.size()+1)));
                    return (pointMap_.size());
                }
            }

            //- Calculation of sign
            inline int sign(double value)
            {
                return (value < 0) ? -1 : ((value > 0) ? 1 : 0);
            }

            //- Store the fetched displacement at present iteration
            inline void storeFetchDisp( double fetchMUIx, 
                                        double fetchMUIy, 
                                        double fetchMUIz,
                                        int pointN)
            {
                fetchDispCoupling_[(3*(pointN-1))+0] = fetchMUIx;
                fetchDispCoupling_[(3*(pointN-1))+1] = fetchMUIy;
                fetchDispCoupling_[(3*(pointN-1))+2] = fetchMUIz;
            }

            //- Calculation of residual at present iteration
            inline void calcResidual(double disPreX,
                                    double disPreY,
                                    double disPreZ,
                                    int pointN)
            {
                residualCoupling_[(3*(pointN-1))+0] = fetchDispCoupling_[(3*(pointN-1))+0] - disPreX;
                residualCoupling_[(3*(pointN-1))+1] = fetchDispCoupling_[(3*(pointN-1))+1] - disPreY;
                residualCoupling_[(3*(pointN-1))+2] = fetchDispCoupling_[(3*(pointN-1))+2] - disPreZ;
            }

            //- Calculation of under relaxation factor at previous iteration
            inline void setUndRelxPrev()
            {
#ifdef USE_MUI
                if (Pstream::master())
                {
                    undRelxCplPrev_ = undRelxCpl_;
                }
                Foam::Pstream::scatter(undRelxCplPrev_);
#else
                if (CppRank_ == 0)
                {
                    undRelxCplPrev_ = undRelxCpl_;
                }
                MPI_Bcast(&undRelxCplPrev_, 1, MPI_DOUBLE, 0, world_);
#endif
            }

            //- Calculation of under relaxation factor at present iteration
            inline void calcUndRelx(int iterN)
            {
                assert(iterN >= 0);
                if (iterN < 2)
                {
                    undRelxCpl_ = initundRelxCpl_;
                }
                else
                {
                    calResidualDelta();
                    double localNominator    = 0.0;
                    double localDenominator  = 0.0;
                    double nominator    = 0.0; 
                    double denominator  = 0.0;
                    localNominator    = std::inner_product(begin(residualCouplingPrev_), 
                                                            end(residualCouplingPrev_), 
                                                            begin(residualDelta_), 0.0);
                    localDenominator  = std::inner_product(begin(residualDelta_), 
                                                            end(residualDelta_), 
                                                            begin(residualDelta_), 0.0); 
#ifdef USE_MUI
                    nominator = localNominator;
                    denominator = localDenominator;
                    Foam::reduce(nominator, sumOp<scalar>());
                    Foam::reduce(denominator, sumOp<scalar>());
                    Foam::Pstream::scatter(nominator);
                    Foam::Pstream::scatter(denominator);
                    std::cout  << "iteration: " << iterN << " Rank: " << Pstream::myProcNo() 
                        << " nominator: " << nominator << " denominator: " << denominator 
                        << " undRelxCplPrev_: " << undRelxCplPrev_ << std::endl;
#else
                    MPI_Allreduce(&localNominator, &nominator, 1, MPI_DOUBLE, MPI_SUM, world_);
                    MPI_Allreduce(&localDenominator, &denominator, 1, MPI_DOUBLE, MPI_SUM, world_);
                    std::cout  << "iteration: " << iterN << " Rank: " << CppRank_ << " nominator: " 
                        << nominator << " denominator: " << denominator << " undRelxCplPrev_: " 
                        << undRelxCplPrev_ << std::endl;
#endif

#ifdef USE_MUI
                    if (Pstream::master())
                    {
                        if (denominator != 0.0 )
                        {
                            if (undRelxCplPrev_ != 0.0)
                            {
                                undRelxCpl_ = -undRelxCplPrev_ * (nominator/denominator);
                            }
                            else
                            {
                                undRelxCpl_ = -initundRelxCpl_ * (nominator/denominator);
                            }
                            // Constraint of the Aitken under relaxation factor
                            undRelxCpl_ = sign(undRelxCpl_) * std::min(std::abs(undRelxCpl_),undRelxCplMax_);
                        }
                        else
                        {
                            assert(nominator == 0);
                            undRelxCpl_ = 0.0;
                        }
                        assert(!isnan( undRelxCpl_));
                        std::cout << "Aitken under relaxation factor is: " << undRelxCpl_ << std::endl;
                    }
                    Foam::Pstream::scatter(undRelxCpl_); 
                    std::cout << "undRelxCpl_: " << undRelxCpl_ << " rank: " << Pstream::myProcNo() << std::endl;
#else
                    if (CppRank_ == 0)
                    {
                        if (denominator != 0.0 )
                        {
                            if (undRelxCplPrev_ != 0.0)
                            {
                                undRelxCpl_ = -undRelxCplPrev_ * (nominator/denominator);
                            }
                            else
                            {
                                undRelxCpl_ = -initundRelxCpl_ * (nominator/denominator);
                            }
                            // Constraint of the Aitken under relaxation factor
                            undRelxCpl_ = sign(undRelxCpl_) * std::min(std::abs(undRelxCpl_),undRelxCplMax_);
                        }
                        else
                        {
                            assert(nominator == 0);
                            undRelxCpl_ = 0.0;
                        }
                        assert(!isnan( undRelxCpl_));
                        std::cout << "Aitken under relaxation factor is: " << undRelxCpl_ << std::endl;
                    }
                    MPI_Bcast(&undRelxCpl_, 1, MPI_DOUBLE, 0, world_);
                    std::cout << "undRelxCpl_: " << undRelxCpl_ << " rank: " << CppRank_ << std::endl;
#endif
                }
            }

            //- Calculation of delta displacement by using Aitken method
            inline void calcDeltaDispAitken()
            {
                for (int pointN=0; pointN<=(3*pointSize_-1); ++pointN)
                {
                    deltaDisp_[pointN] = undRelxCpl_ * residualCoupling_[pointN];
                }
            }

            //- Calculation of delta residual at this iteration
            inline void calResidualDelta()
            {
                for (int pointN=0; pointN<=(3*pointSize_-1); ++pointN)
                {
                    residualDelta_[pointN] = residualCoupling_[pointN] - residualCouplingPrev_[pointN];
                }
            }

            //- Store residual at this iteration
            inline void setPrevResidual()
            {
                for (int pointN=0; pointN<=(3*pointSize_-1); ++pointN)
                {
                    residualCouplingPrev_[pointN] = residualCoupling_[pointN];
                }
            }

            //- Reset of residualMagSqSum_
            inline void resetResidualMagSqSum()
            {
                residualMagSqSum_ = 0.0;
            }

            //- Accumulation of residualMagSqSum_
            inline void accumResidualMagSqSum()
            {
                double localResidualMagSqSum_ = 0.0;
                for (int pointN=0; pointN<=(3*pointSize_-1); ++pointN)
                {
                    localResidualMagSqSum_ += pow(residualCoupling_[pointN], 2);
                }

#ifdef USE_MUI
                std::cout << "{MUI_Coupling} From rank " << Pstream::myProcNo() 
                    << " localResidualMagSqSum_: " << localResidualMagSqSum_ << std::endl;
                residualMagSqSum_ = localResidualMagSqSum_;
                Foam::reduce(residualMagSqSum_, sumOp<scalar>());
                Foam::Pstream::scatter(residualMagSqSum_);
#else
                std::cout << "{MUI_Coupling} From rank " << CppRank_ << " localResidualMagSqSum_: " 
                << localResidualMagSqSum_ << std::endl;
                MPI_Allreduce(&localResidualMagSqSum_, &residualMagSqSum_, 1, MPI_DOUBLE, MPI_SUM, world_);
#endif
            }

            //- Calculation of residualL2Norm_
            inline void calResidualL2Norm()
            {
#ifdef USE_MUI
                if(Pstream::master())
                {
                    residualL2Norm_ = std::sqrt(residualMagSqSum_);
                }
                Foam::Pstream::scatter(residualL2Norm_);
                std::cout << "residualMagSqSum_: " << residualMagSqSum_ 
                    << " residualL2Norm_: " << residualL2Norm_ << " rank: " 
                    << Pstream::myProcNo() << std::endl;
#else
                if (CppRank_ == 0)
                {
                    residualL2Norm_ = std::sqrt(residualMagSqSum_);
                }
                MPI_Bcast(&residualL2Norm_, 1, MPI_DOUBLE, 0, world_);
                std::cout << "residualMagSqSum_: " << residualMagSqSum_ << " residualL2Norm_: " 
                    << residualL2Norm_ << " rank: " << CppRank_ << std::endl;
#endif
            }

            //- Calculation of residualL2NormMax_
            inline void calResidualL2NormMax()
            {
#ifdef USE_MUI
                if(Pstream::master())
                {
                    residualL2NormMax_ = max(residualL2Norm_, residualL2NormMax_);
                }       
                Foam::Pstream::scatter(residualL2NormMax_);
                std::cout << "residualL2NormMax_: " << residualL2NormMax_ << " residualL2Norm_: " 
                    << residualL2Norm_ << " rank: " << Pstream::myProcNo() << std::endl;
#else
                if (CppRank_ == 0)
                {
                    residualL2NormMax_ = std::max(residualL2Norm_, residualL2NormMax_);
                }
                MPI_Bcast(&residualL2NormMax_, 1, MPI_DOUBLE, 0, world_);
                std::cout << "residualL2NormMax_: " << residualL2NormMax_ << " residualL2Norm_: " 
                    << residualL2Norm_ << " rank: " << CppRank_ << std::endl;
#endif
            }

            //- Correction of residualL2Norm_ based on the maximum residualL2NormMax_
            inline void setResidualL2Norm()
            {
#ifdef USE_MUI
                if(Pstream::master())
                {
                    if (residualL2NormMax_ <= 0.0 )
                    {
                        residualL2Norm_ = 1.0;
                    }
                    else
                    {
                        residualL2Norm_ /= residualL2NormMax_;
                    }
                }
                Foam::Pstream::scatter(residualL2Norm_);
#else
                if (CppRank_ == 0)
                {
                    if (residualL2NormMax_ <= 0.0 )
                    {
                        residualL2Norm_ = 1.0;
                    }
                    else
                    {
                        residualL2Norm_ /= residualL2NormMax_;
                    }
                }
                MPI_Bcast(&residualL2Norm_, 1, MPI_DOUBLE, 0, world_);
#endif
            }

            inline void reSetRefMatrixes()
            {
                if (globalAlpha)
                {
#ifdef USE_MUI
                    if(Pstream::master())
                    {
                        matrixVref.resize(0,0);
                    }
#else
                    if (CppRank_ == 0)
                    {
                        matrixVref.resize(0,0);
                    }
#endif
                }
                else
                {
                    matrixVref.resize(0,0); 
                }
                matrixWref.resize(0,0);
            }

            inline void calMatrixes(int iterN)
            {

                // Calculate matrixVk
                if (globalAlpha)
                {
#ifdef USE_MUI
                    if(Pstream::master())
                    {
                        matrixVk.resize(0,0);
                        Eigen::VectorXd vectorVTemp = Eigen::VectorXd::Map(&globalResidualCoupling_[0],globalResidualCoupling_.size());
                        matrixVk.resize(globalResidualCouplingSizeSum_,iterN-1);
                        matrixVk = matrixVref.array().colwise() - vectorVTemp.array();
                        vectorVTemp.resize(0);
                    }
#else
                    if (CppRank_ == 0)
                    {
                        matrixVk.resize(0,0);
                        Eigen::VectorXd vectorVTemp = Eigen::VectorXd::Map(&globalResidualCoupling_[0],globalResidualCoupling_.size());
                        matrixVk.resize(globalResidualCouplingSizeSum_,iterN-1);
                        matrixVk = matrixVref.array().colwise() - vectorVTemp.array();
                        vectorVTemp.resize(0);
                    }
#endif
                }
                else
                {
                    matrixVk.resize(0,0);
                    Eigen::VectorXd vectorVTemp = Eigen::VectorXd::Map(&residualCoupling_[0],residualCoupling_.size());
                    matrixVk.resize(3*pointSize_,iterN-1);
                    matrixVk = matrixVref.array().colwise() - vectorVTemp.array();
                    vectorVTemp.resize(0);
                }

                // Calculate matrixWk
                matrixWk.resize(0,0);
                Eigen::VectorXd vectorWTemp = Eigen::VectorXd::Map(&fetchDispCoupling_[0],fetchDispCoupling_.size());
                matrixWk.resize(3*pointSize_,iterN-1);
                matrixWk = matrixWref.array().colwise() - vectorWTemp.array();
                vectorWTemp.resize(0);
            }

            inline void refMatrixesPushback(int iterN)
            {
                // Push back matrixVref
                if (globalAlpha)
                {
#ifdef USE_MUI
                    if(Pstream::master())
                    {
                        Eigen::MatrixXd matrixVrefTemp = matrixVref;
                        Eigen::MatrixXd vectorVrefTemp = Eigen::VectorXd::Map(&globalResidualCoupling_[0],globalResidualCoupling_.size());
                        matrixVref.resize(globalResidualCouplingSizeSum_,iterN);
                        if(matrixVrefTemp.size() != 0)
                        {
                            matrixVref.leftCols(iterN-1) = matrixVrefTemp;
                        }
                        matrixVref.rightCols(1) = vectorVrefTemp.rightCols(1);
                        matrixVrefTemp.resize(0,0);
                        vectorVrefTemp.resize(0,0);
                    }
#else
                    if (CppRank_ == 0)
                    {
                        Eigen::MatrixXd matrixVrefTemp = matrixVref;
                        Eigen::MatrixXd vectorVrefTemp = Eigen::VectorXd::Map(&globalResidualCoupling_[0],globalResidualCoupling_.size());
                        matrixVref.resize(globalResidualCouplingSizeSum_,iterN);
                        if(matrixVrefTemp.size() != 0)
                        {
                            matrixVref.leftCols(iterN-1) = matrixVrefTemp;
                        }
                        matrixVref.rightCols(1) = vectorVrefTemp.rightCols(1);
                        matrixVrefTemp.resize(0,0);
                        vectorVrefTemp.resize(0,0);
                    }
#endif
                }
                else
                {
                    Eigen::MatrixXd matrixVrefTemp = matrixVref;
                    Eigen::MatrixXd vectorVrefTemp = Eigen::VectorXd::Map(&residualCoupling_[0],residualCoupling_.size());
                    matrixVref.resize(3*pointSize_,iterN);
                    if(matrixVrefTemp.size() != 0)
                    {
                        matrixVref.leftCols(iterN-1) = matrixVrefTemp;
                    }
                    matrixVref.rightCols(1) = vectorVrefTemp.rightCols(1);
                    matrixVrefTemp.resize(0,0);
                    vectorVrefTemp.resize(0,0); 
                }

                // Push back matrixWref
                Eigen::MatrixXd matrixWrefTemp = matrixWref;
                Eigen::MatrixXd vectorWrefTemp = Eigen::VectorXd::Map(&fetchDispCoupling_[0],fetchDispCoupling_.size());
                matrixWref.resize(3*pointSize_,iterN);
                if(matrixWrefTemp.size() != 0)
                {
                    matrixWref.leftCols(iterN-1) = matrixWrefTemp;
                }
                matrixWref.rightCols(1) = vectorWrefTemp.rightCols(1);
                matrixWrefTemp.resize(0,0);
                vectorWrefTemp.resize(0,0);
            }

            inline void qRDecomposition()
            {
                if (globalAlpha)
                {
#ifdef USE_MUI
                    if(Pstream::master())
                    {
                        matrixR.resize(0,0);
                        matrixQ.resize(0,0);
                        Eigen::HouseholderQR<Eigen::MatrixXd> qr;
                        qr.compute(matrixVk);
                        matrixR = qr.matrixQR().triangularView<Eigen::Upper>();
                        matrixQ = qr.householderQ();
                    }
#else
                    if (CppRank_ == 0)
                    {
                        matrixR.resize(0,0);
                        matrixQ.resize(0,0);
                        Eigen::HouseholderQR<Eigen::MatrixXd> qr;
                        qr.compute(matrixVk);
                        matrixR = qr.matrixQR().triangularView<Eigen::Upper>();
                        matrixQ = qr.householderQ();
                    }
#endif
                }
                else
                {
                        matrixR.resize(0,0);
                        matrixQ.resize(0,0);
                        Eigen::HouseholderQR<Eigen::MatrixXd> qr;
                        qr.compute(matrixVk);
                        matrixR = qr.matrixQR().triangularView<Eigen::Upper>();
                        matrixQ = qr.householderQ();        
                }
            }

            inline void calAlpha(int iterN)
            {
                vectorAlpha.resize(iterN-1);
                if (globalAlpha)
                {
#ifdef USE_MUI
                    if(Pstream::master())
                    {
                        Eigen::VectorXd vectorVTemp = Eigen::VectorXd::Map(&globalResidualCoupling_[0],globalResidualCoupling_.size());
                        Eigen::VectorXd vectorBTemp = (-1.) * (matrixQ.transpose()) * vectorVTemp;
                        vectorAlpha = matrixR.fullPivHouseholderQr().solve(vectorBTemp);
                        vectorVTemp.resize(0);
                        vectorBTemp.resize(0);
                    }
                    Foam::List<scalar> vectorAlphaFoamListTemp(vectorAlpha.size());
                    forAll(vectorAlphaFoamListTemp, i)
                    {
                        vectorAlphaFoamListTemp[i] =  vectorAlpha[i];
                    }
                    Foam::Pstream::scatterList(vectorAlphaFoamListTemp);
                    std::vector<double> vectorAlphatemp2;
                    vectorAlphatemp2.resize(vectorAlpha.size());
                    forAll(vectorAlphaFoamListTemp, i)
                    {
                        vectorAlphatemp2[i] =  vectorAlphaFoamListTemp[i];
                    }
                    vectorAlpha = Eigen::VectorXd::Map(vectorAlphatemp2.data(), vectorAlphatemp2.size());
#else
                    if (CppRank_ == 0)
                    {
                        Eigen::VectorXd vectorVTemp = Eigen::VectorXd::Map(&globalResidualCoupling_[0],globalResidualCoupling_.size());
                        Eigen::VectorXd vectorBTemp = (-1.) * (matrixQ.transpose()) * vectorVTemp;
                        vectorAlpha = matrixR.fullPivHouseholderQr().solve(vectorBTemp);
                        vectorVTemp.resize(0);
                        vectorBTemp.resize(0);
                    }
                    MPI_Bcast(  vectorAlpha.data(), 
                                iterN-1, 
                                MPI_DOUBLE, 
                                0, 
                                world_);
#endif
                }
                else
                {
                    Eigen::VectorXd vectorVTemp = Eigen::VectorXd::Map(&residualCoupling_[0],residualCoupling_.size());
                    Eigen::VectorXd vectorBTemp = (-1.) * (matrixQ.transpose()) * vectorVTemp;
                    Eigen::VectorXd vectorAlphaTemp = matrixR.fullPivHouseholderQr().solve(vectorBTemp);
                    vectorVTemp.resize(0);
                    vectorBTemp.resize(0); 
#ifdef USE_MUI
                    for(int i = 0; i < vectorAlphaTemp.size(); i++)
                    {
                        scalar vectorAlphaScalarTemp;
                        vectorAlphaScalarTemp = vectorAlphaTemp(i);
                        Foam::reduce(vectorAlphaScalarTemp, sumOp<scalar>());
                        Foam::Pstream::scatter(vectorAlphaScalarTemp);
                        vectorAlpha(i) = vectorAlphaScalarTemp;
                    }
#else
                    MPI_Allreduce(vectorAlphaTemp.data(), vectorAlpha.data(), (iterN-1), MPI_DOUBLE, MPI_SUM, world_);
#endif
                    vectorAlpha /= CppSize_;
                }
            }

            inline void calcDeltaDispIQN()
            {
                Eigen::VectorXd vectorDeltaDispTemp = matrixWk * vectorAlpha;
                Eigen::VectorXd::Map(&deltaDisp_[0], vectorDeltaDispTemp.size()) = vectorDeltaDispTemp;
                vectorDeltaDispTemp.resize(0);
            }

            inline void calLocalResidualSize()
            {
                int localResidualCouplingSize = residualCoupling_.size();
#ifdef USE_MUI
                Foam::List<scalar> globalResidualCouplingSizeFoamListTemp(Pstream::nProcs());
                forAll(globalResidualCouplingSizeFoamListTemp, i)
                {
                    if(Pstream::myProcNo() == i)
                    {
                        globalResidualCouplingSizeFoamListTemp[i] =  localResidualCouplingSize;
                    }
                }
                Foam::Pstream::gatherList(globalResidualCouplingSizeFoamListTemp);
                Foam::Pstream::scatterList(globalResidualCouplingSizeFoamListTemp);
                forAll(globalResidualCouplingSizeFoamListTemp, i)
                {
                   globalResidualCouplingSize_[i] =  globalResidualCouplingSizeFoamListTemp[i];
                }
                globalResidualCouplingSizeSum_ = localResidualCouplingSize;
                Foam::reduce(globalResidualCouplingSizeSum_, sumOp<scalar>());
                Foam::Pstream::scatter(globalResidualCouplingSizeSum_);
#else
                MPI_Gather( &localResidualCouplingSize, 
                            1, 
                            MPI_INT, 
                            globalResidualCouplingSize_.data(), 
                            1, 
                            MPI_INT, 
                            0, 
                            world_);
                MPI_Bcast(  globalResidualCouplingSize_.data(), 
                            CppSize_, 
                            MPI_INT, 
                            0, 
                            world_);
                MPI_Allreduce(  &localResidualCouplingSize, 
                                &globalResidualCouplingSizeSum_, 
                                1, 
                                MPI_INT, 
                                MPI_SUM, 
                                world_);
                MPI_Bcast(  &globalResidualCouplingSizeSum_, 
                            1, 
                            MPI_INT, 
                            0, 
                            world_);
#endif
            }
            
            inline void gatherLocalResidual()
            {
#ifdef USE_MUI
                if (Pstream::master())
                {
                    globalResidualCoupling_.resize(globalResidualCouplingSizeSum_, 0.0);
                }
                Foam::List<scalar> globalResidualCouplingFoamListTemp(residualCoupling_.size());
                forAll(globalResidualCouplingFoamListTemp, i)
                {
                   globalResidualCouplingFoamListTemp[i] =  residualCoupling_[i];
                }
                Foam::Pstream::gatherList(globalResidualCouplingFoamListTemp);
                Foam::Pstream::scatterList(globalResidualCouplingFoamListTemp);
                forAll(globalResidualCouplingFoamListTemp, i)
                {
                   globalResidualCoupling_[i] =  globalResidualCouplingFoamListTemp[i];
                }
                if (Pstream::master())
                {
                    assert(globalResidualCoupling_.size() == globalResidualCouplingSizeSum_);
                }
#else
                if (CppRank_ == 0)
                {
                    globalResidualCoupling_.resize(globalResidualCouplingSizeSum_, 0.0);
                }
                MPI_Gather( residualCoupling_.data(), 
                            globalResidualCouplingSize_[CppRank_], 
                            MPI_DOUBLE,
                            globalResidualCoupling_.data(), 
                            globalResidualCouplingSize_[CppRank_], 
                            MPI_DOUBLE, 
                            0, 
                            world_);
                if (CppRank_ == 0)
                {
                    assert(globalResidualCoupling_.size() == globalResidualCouplingSizeSum_);
                }
#endif
            }

            inline void freeGatherResidual()
            {
#ifdef USE_MUI
                if (Pstream::master())
                {
                    std::vector<double>().swap(globalResidualCoupling_);
                }
#else
                if (CppRank_ == 0)
                {
                    std::vector<double>().swap(globalResidualCoupling_);
                }
#endif
            }

            //- Logging
            inline void logging()
            {
#ifdef USE_MUI
                if(Pstream::master())
                {
                    std::cout << std::scientific;
                    std::cout << std::endl;
                    std::cout << "{MUI_Coupling} No. of points is: " << pointSize() << std::endl;
                    std::cout << "{MUI_Coupling} Under-relaxation factor is: " << undRelxCpl() << std::endl;
                    std::cout << "{MUI_Coupling} Size of residualCoupling_ is: " << residualCoupling_.size() << std::endl;
                    std::cout << "{MUI_Coupling} Size of residualCouplingPrev_ is: " << residualCouplingPrev_.size() << std::endl;
                    std::cout << "{MUI_Coupling} Size of residualMagSq_ is: " << residualMagSq_.size() << std::endl;
                    std::cout << std::endl;
                    std::cout << "{MUI_Coupling} Residual of MUI Coupling is: " << residualL2Norm_ << std::endl;
                    std::cout << "{MUI_Coupling} residualL2NormMax of MUI Coupling is: " << residualL2NormMax_ << std::endl;
                    std::cout << "{MUI_Coupling} residualMagSqSum of MUI Coupling is: " << residualMagSqSum_ << std::endl;
                }              
#else
                if (CppRank_ == 0)
                {
                    std::cout << std::scientific;
                    std::cout << std::endl;
                    std::cout << "{MUI_Coupling} No. of points is: " << pointSize() << std::endl;
                    std::cout << "{MUI_Coupling} Under-relaxation factor is: " << undRelxCpl() << std::endl;
                    std::cout << "{MUI_Coupling} Size of residualCoupling_ is: " << residualCoupling_.size() << std::endl;
                    std::cout << "{MUI_Coupling} Size of residualCouplingPrev_ is: " << residualCouplingPrev_.size() << std::endl;
                    std::cout << "{MUI_Coupling} Size of residualMagSq_ is: " << residualMagSq_.size() << std::endl;
                    std::cout << std::endl;
                    std::cout << "{MUI_Coupling} Residual of MUI Coupling is: " << residualL2Norm_ << std::endl;
                    std::cout << "{MUI_Coupling} residualL2NormMax of MUI Coupling is: " << residualL2NormMax_ << std::endl;
                    std::cout << "{MUI_Coupling} residualMagSqSum of MUI Coupling is: " << residualMagSqSum_ << std::endl;
                }
#endif
            }

    protected:

        // Protected member functions

    public:
        // Static data members

        // Constructors

            //- Constructor - default
            inline muiCouplingIQNILS
            (
                double initUndRelxCpl
            ):
            initundRelxCpl_(initUndRelxCpl)
            {

                std::cout << "{MUI_Coupling} C++ IQNILS default constructor" 
                    << " initial under-relaxation factor: " << initundRelxCpl_<< std::endl;
            }

            //- Construct from components
            inline muiCouplingIQNILS
            (
                int pointSize,
                double initUndRelxCpl,
                double undRelxCplMax = 1.0,
                int aitkenIterationN = 5,
                bool globalAlphaInput = false
            )
            : 
                pointSize_(pointSize),
                initundRelxCpl_(initUndRelxCpl),
                undRelxCplMax_(undRelxCplMax),
                aitkenIterationN_(aitkenIterationN),
                globalAlpha(globalAlphaInput)
            {

                std::cout << "{MUI_Coupling} C++ IQNILS constructor: " 
                    <<" number of points: " << pointSize_ 
                    << " initial under-relaxation factor: " << initundRelxCpl_ 
                    << " numbers of Aitken Iterations per IQN-ILS time-step: " << aitkenIterationN 
                    << " globalAlpha: " << globalAlpha << std::endl;
				
				world_ = MPI_COMM_WORLD;
				
				MPI_Comm_rank( world_, &CppRank_ );
                MPI_Comm_size( world_, &CppSize_ );
                std::cout << "{MUI_Coupling} Hello world from rank: " << CppRank_ << " out of " 
                    << CppSize_ << " processors. " <<std::endl;
            }

            //- Construct from components
            inline muiCouplingIQNILS
            (
                int pointSize,
                double initUndRelxCpl,
                MPI_Comm *Cppworld,
                double undRelxCplMax = 1.0,
                int aitkenIterationN = 5,
                bool globalAlphaInput = false
            )
            : 
                pointSize_(pointSize),
                initundRelxCpl_(initUndRelxCpl),
                world_(*Cppworld),
                undRelxCplMax_(undRelxCplMax),
                aitkenIterationN_(aitkenIterationN),
                globalAlpha(globalAlphaInput)
            {
                std::cout << "{MUI_Coupling} C++ IQNILS constructor: "
                    <<" number of points: " << pointSize_ 
                    << " initial under-relaxation factor: " << initundRelxCpl_ 
                    << " numbers of Aitken Iterations per IQN-ILS time-step: " << aitkenIterationN
                    << " globalAlpha: " << globalAlpha <<std::endl;
                MPI_Comm_rank( world_, &CppRank_ );
                MPI_Comm_size( world_, &CppSize_ );
                std::cout << "{MUI_Coupling} Hello world from rank: " << CppRank_ << " out of " 
                    << CppSize_ << " processors. " <<std::endl;
            }
             //- Construct from components and create MPI split
            inline muiCouplingIQNILS
            (
                int pointSize,
                double initUndRelxCpl,
                std::vector<int> &localRankVec,
				int sizeAfterSplit,
                double undRelxCplMax = 1.0,
                int aitkenIterationN = 5,
                bool globalAlphaInput = false
            )
            : 
                pointSize_(pointSize),
                initundRelxCpl_(initUndRelxCpl),
                localRankVec_(localRankVec),
                sizeAfterSplit_(sizeAfterSplit),
                undRelxCplMax_(undRelxCplMax),
                aitkenIterationN_(aitkenIterationN),
                globalAlpha(globalAlphaInput)
            {
                std::cout << "{MUI_Coupling} C++ IQNILS constructor: "
                    <<" number of points: " << pointSize_ 
                    << " initial under-relaxation factor: " << initundRelxCpl_ 
                    << " numbers of Aitken Iterations per IQN-ILS time-step: " << aitkenIterationN
                    << " globalAlpha: " << globalAlpha <<std::endl;

				MPI_Comm_rank( MPI_COMM_WORLD, &CppTotalRank_ );
                MPI_Comm_size( MPI_COMM_WORLD, &CppTotalSize_ );
                std::cout << "{MUI_Coupling} Total rank: " << CppTotalRank_ << " out of total size: " 
                    << CppTotalSize_ << " processors. " <<std::endl;

                const int *localRanksArray=&localRankVec_[0];

				// Get the group of processes in MPI_COMM_WORLD
				MPI_Group world_group;
				MPI_Comm_group(MPI_COMM_WORLD, &world_group);

 				MPI_Group localGroup;
 				MPI_Group_incl(world_group, sizeAfterSplit_, localRanksArray, &localGroup);
 				MPI_Comm localComm;
				MPI_Comm_create_group(MPI_COMM_WORLD, localGroup, 0, &localComm);

				world_ = localComm;

                MPI_Comm_rank( world_, &CppRank_ );
                MPI_Comm_size( world_, &CppSize_ );
				if (sizeAfterSplit_ == CppSize_)
				{
					std::cout << "{MUI_Coupling} Total size: " << CppRank_ << " out of " 
						<< CppSize_ << " processors. " <<std::endl;
				} else{
					std::cerr << "sizeAfterSplit: " << sizeAfterSplit << " doesnot equals to the size " 
					<< CppSize_ <<std::endl;
				}
            }
  
        // Selectors

        // Destructor - default

            inline ~muiCouplingIQNILS
            (
        
            )
            {
                std::cout << "{MUI_Coupling} C++ IQNILS destructor" << std::endl;
            }

        // Member Functions
            // Access

                //- Return under relaxation factor of the coupling
                inline double undRelxCpl()
                {
                    return undRelxCpl_;
                }

                //- Return No. of points
                inline int pointSize() const
                {
                    return pointSize_;
                }

                //- Return square sum of the residual
                inline double residualMagSqSum() const
                {
                    return residualMagSqSum_;
                }

                //- Return maximum value of the residual L-2 Norm
                inline double residualL2NormMax() const
                {
                    return residualL2NormMax_;
                }

                //- Return the value of the residual L-2 Norm
                inline double residualL2Norm() const
                {
                    return residualL2Norm_;
                }

                //- Return x axis component of the delta displacement
                inline double getXDeltaDisp(int pointV)
                {
                    it_ = pointMap_.find(pointV);
                    assert (it_ != pointMap_.end());
                    return deltaDisp_[(3*(((*it_).second)-1))+0];
                }

                //- Return y axis component of the delta displacement
                inline double getYDeltaDisp(int pointV)
                {
                    it_ = pointMap_.find(pointV);
                    assert (it_ != pointMap_.end());
                    return deltaDisp_[(3*(((*it_).second)-1))+1];
                }

                //- Return z axis component of the delta displacement
                inline double getZDeltaDisp(int pointV)
                {
                    it_ = pointMap_.find(pointV);
                    assert (it_ != pointMap_.end());
                    return deltaDisp_[(3*(((*it_).second)-1))+2];
                }

              // Edit

                //- Initialize coupling method
                inline void initialize(int pointSize,
                                       double undRelxCplMax = 1.0,
                                       int aitkenIterationN = 5,
                                       bool globalAlphaInput = false)
                {
                    pointSize_ = pointSize;
                    undRelxCplMax_ = undRelxCplMax;
                    aitkenIterationN_ = aitkenIterationN;
                    globalAlpha = globalAlphaInput;
                    residualMagSq_.resize(pointSize_, 0.0);
                    residualDelta_.resize((3*pointSize_), 0.0);
                    residualCoupling_.resize((3*pointSize_), 0.0);
                    residualCouplingPrev_.resize((3*pointSize_), 0.0);
                    fetchDispCoupling_.resize((3*pointSize_), 0.0);
                    deltaDisp_.resize((3*pointSize_), 0.0);
                    if (globalAlpha)
                    {
                        needCalLocalResidualSize = true;
                        globalResidualCouplingSize_.resize(CppSize_, 0);
                    }
                    calcUndRelx(0);
                    setUndRelxPrev();
                    std::cout << "{MUI_Coupling} C++ IQNILS initialize: "
                        <<" number of points: " << pointSize_
                        << " numbers of Aitken Iterations per IQN-ILS time-step: " << aitkenIterationN
                        << " globalAlpha" << globalAlpha <<std::endl;
                }
 
                //- Initialize coupling method
                inline void initialize(int pointSize,
                                       MPI_Comm *Cppworld,
                                       double undRelxCplMax = 1.0,
                                       int aitkenIterationN = 5,
                                       bool globalAlphaInput = false)
                {
                    pointSize_ = pointSize;
                    world_ = *Cppworld;
                    undRelxCplMax_ = undRelxCplMax;
                    aitkenIterationN_ = aitkenIterationN;
                    globalAlpha = globalAlphaInput;
                    residualMagSq_.resize(pointSize_, 0.0);
                    residualDelta_.resize((3*pointSize_), 0.0);
                    residualCoupling_.resize((3*pointSize_), 0.0);
                    residualCouplingPrev_.resize((3*pointSize_), 0.0);
                    fetchDispCoupling_.resize((3*pointSize_), 0.0);
                    deltaDisp_.resize((3*pointSize_), 0.0);
                    if (globalAlpha)
                    {
                        needCalLocalResidualSize = true;
                        globalResidualCouplingSize_.resize(CppSize_, 0);
                    }
                    calcUndRelx(0);
                    setUndRelxPrev();
                    std::cout << "{MUI_Coupling} C++ IQNILS initialize: "
                        <<" number of points: " << pointSize_
                        << " numbers of Aitken Iterations per IQN-ILS time-step: " << aitkenIterationN
                        << " globalAlpha" << globalAlpha <<std::endl;
                    MPI_Comm_rank( world_, &CppRank_ );
                    MPI_Comm_size( world_, &CppSize_ );
                    std::cout << "{MUI_Coupling} Hello world from rank: " << CppRank_ << " out of " 
                        << CppSize_ << " processors. " <<std::endl;
                }
 
                //- Initialize coupling method - overload
                inline void initialize()
                {
                    residualMagSq_.resize(pointSize_, 0.0);
                    residualDelta_.resize((3*pointSize_), 0.0);
                    residualCoupling_.resize((3*pointSize_), 0.0);
                    residualCouplingPrev_.resize((3*pointSize_), 0.0);
                    fetchDispCoupling_.resize((3*pointSize_), 0.0);
                    deltaDisp_.resize((3*pointSize_), 0.0);
                    if (globalAlpha)
                    {
                        needCalLocalResidualSize = true;
                        globalResidualCouplingSize_.resize(CppSize_, 0);
                    }
                    calcUndRelx(0);
                    setUndRelxPrev();
                }

                //- Collection of Residual at this iteration
                inline void collectResidual(double fetchMUIx,
                                            double fetchMUIy,
                                            double fetchMUIz,
                                            double disPreX,
                                            double disPreY,
                                            double disPreZ,
                                            int pointV)
                {
                    //calculate point ID
                    int pointN = -999;
                    pointN = calcpointN(pointV);
                    assert (pointN >= 0);
                    storeFetchDisp( fetchMUIx,
                                    fetchMUIy,
                                    fetchMUIz,
                                    pointN);
                    calcResidual(   disPreX,
                                    disPreY,
                                    disPreZ,
                                    pointN);
                }

                //- Collection of coupling process at this iteration
                inline void process(int iterN)
                {
                    assert(iterN >= 0);
                    if (globalAlpha)
                    {
                        if(needCalLocalResidualSize)
                        {
                            calLocalResidualSize();
                            needCalLocalResidualSize = false;
                        }
                    }
                    if (iterN == 0)
                    {
                    }
                    else if ((iterN <= aitkenIterationN_) && (iterN > 0))
                    {
                        if (iterN == 1)
                        {
                            reSetRefMatrixes();
                        }
                        if (globalAlpha)
                        {
                            gatherLocalResidual();
                        }
                        calcUndRelx(iterN);
                        calcDeltaDispAitken();
                        setPrevResidual();
                        accumResidualMagSqSum();
                        calResidualL2Norm();
                        calResidualL2NormMax();
                        setResidualL2Norm();
                        setUndRelxPrev();
                        logging();
                        resetResidualMagSqSum();
                        refMatrixesPushback(iterN);
                        if (globalAlpha)
                        {
                            freeGatherResidual();
                        }
                    }
                    else
                    {
                        if (globalAlpha)
                        {
                            gatherLocalResidual();
                        }
                        calMatrixes(iterN);
                        qRDecomposition();
                        calAlpha(iterN);
                        calcDeltaDispIQN();
                        refMatrixesPushback(iterN);
                        setPrevResidual();
                        accumResidualMagSqSum();
                        calResidualL2Norm();
                        calResidualL2NormMax();
                        setResidualL2Norm();
                        setUndRelxPrev();
                        logging();
                        if (globalAlpha)
                        {
                            resetResidualMagSqSum();
                        }
                    }
                }
    };

    // Global Functions

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace muiCoupling

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif // End muiCouplingIQNILS_HPP

// ************************************************************************* //