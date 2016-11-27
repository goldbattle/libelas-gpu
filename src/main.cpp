//
//  main.cpp
//  disparityMap
//
//  Created by Awais Aslam on 04/05/16.
//  https://github.com/awais-aslam-coder/stereo-disparitymap-c
//
//

#include <iostream>
#include "lodepng.h"
#include <fstream>
#include "math.h"
#include <sys/time.h>
#include <ctime>

using namespace std;

int imW;
int imH;
int rW;
int rH;
int mapW;
int mapH;

unsigned char* im0;
unsigned char* im1;

float **imGray0;
float **imGray1;

int **dispMapL;
int **dispMapR;
int **dispMap;
int **dispMapF;

const int winX = 25;
const int winY = 25;
const int threshold = 5;

const int ndisp = 260;
const int scaleFactor = 4;
const int dMin = 0;
const int dMax = ndisp / scaleFactor;

const float RC = 0.2126;
const float GC = 0.7152;
const float BC = 0.0722;

int **normalizeImage(int **image, int maxPixelValue);

int loadImages();
void init();
void resizeAndGrayScale();
void computeDisparityMap();
void crossCheck();
void doOcclusionFilling();

void outputGrayScaleImage(float **image, const char *filename);
void outputImage(int **image, const char *filename);

double findExecutionTime(timeval t1,timeval t2);

int main(int argc, const char * argv[]) {
    
    timeval t1,t2,start,stop;
    
    printf("      ------ Starting Application ------\n\n");
    
    gettimeofday(&start,NULL);
    
    gettimeofday(&t1,NULL);
    
    if(loadImages()){
        if(loadImages()){
        } else {
            printf("Application has Failed to load images.. Aborting... \n");
            return 0;
        }
    }
    
    gettimeofday(&t2,NULL);
    printf("1) Image Load & Decoding Completed --- %lf seconds.\n",findExecutionTime(t1,t2));
    printf("===> Next Step : Memory Allocation & Init Process ....\n");
    
    gettimeofday(&t1,NULL);
    init();
    gettimeofday(&t2,NULL);
    printf("--- Allocating Memory and Initializing Matrices --- %lf seconds.\n",findExecutionTime(t1,t2));
    printf("===> Next Step : Gray Scale Conversion ....\n");
    
    gettimeofday(&t1,NULL);
    
    //////////////////////////////////////////////////////
    
    resizeAndGrayScale();
    
    outputGrayScaleImage(imGray0,"../output/im0Gray.png");
    outputGrayScaleImage(imGray1,"../output/im1Gray.png");
    
    gettimeofday(&t2,NULL);
    
    printf("2) Gray Scale Conversion Completed --- %lf seconds.\n",findExecutionTime(t1, t2));
    printf("===> Next Step : Disparity Map Computation ....\n");
    
    
    /////////////////////////////////////////////////////
    
    gettimeofday(&t1,NULL);
    
    computeDisparityMap();
    
    outputImage(normalizeImage(dispMapR,255),"../output/dispMapRight.png");
    outputImage(normalizeImage(dispMapL,255),"../output/dispMapLeft.png");
    
    gettimeofday(&t2,NULL);
    
    printf("3) Disparity Map Computation Completed --- %lf seconds.\n",findExecutionTime(t1, t2));
    printf("===> Next Step : Cross Check ....\n");
    
    ////////////////////////////////////////////////////
    
    gettimeofday(&t1,NULL);
    
    crossCheck();
    
    outputImage(normalizeImage(dispMap,255),"../output/dispMap1.png");
    
    gettimeofday(&t2,NULL);
    printf("4) Crosscheck Post Processing Step Completed. --- %lf seconds.\n",findExecutionTime(t1, t2));
    printf("===> Next Step : Occlusion Filling ....\n");
    
    ////////////////////////////////////////////////////
    
    gettimeofday(&t1,NULL);
    
    doOcclusionFilling();
    
    outputImage(normalizeImage(dispMapF,255),"../output/dispMap2.png");
    
    gettimeofday(&t2,NULL);
    
    printf("4) Occlusion Filling Post Processing Step Completed. --- %lf seconds.\n",findExecutionTime(t1, t2));
    printf("===> Steps Finished :) \n");
    ///////////////////////////////////////////////////
    
    
    gettimeofday(&stop,NULL);
    printf("\nTotal Execution Time : %lf seconds\n\n",findExecutionTime(start, stop));
    
    
    printf("      ------ Ending Application ------\n");
    
    return 1;
}

double findExecutionTime(timeval start,timeval end){
    return (end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000;
}

int loadImages(){
    
    unsigned w0,h0,w1,h1;
    unsigned error;
    
    // Read and Decode im0.png
    error = lodepng_decode32_file(&im0, &w0, &h0, "../images/im0.png");
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 0;
    }
    
    // Read and Decode im1.png
    error = lodepng_decode32_file(&im1, &w1, &h1, "../images/im1.png");
    if (error) {
        printf("error %u: %s\n", error, lodepng_error_text(error));
        return 0;
    }
    
    imW = w0;
    imH = h0;
    
    rW = imW / scaleFactor;
    rH = imH / scaleFactor;
    
    mapW = rW + (winX - 1) * 2 + dMax * 2;
    mapH = rH + (winY - 1) * 2;
    
    return 1;
}

void init(){
    
    imGray0 = new float *[mapW];
    imGray1 = new float *[mapW];
    dispMapL = new int *[mapW];
    dispMapR = new int *[mapW];
    dispMap = new int *[mapW];
    dispMapF = new int *[mapW];
    
    for(int i = 0; i < mapW; i++){
        imGray0[i] = new float [mapH];
        imGray1[i] = new float [mapH];
        dispMapL[i] = new int [mapH];
        dispMapR[i] = new int [mapH];
        dispMap[i] = new int [mapH];
        dispMapF[i] = new int [mapH];
    }
    
    for(int i =0; i < mapW; i++){
        for(int j = 0; j < mapH; j++){
            imGray0[i][j] = 0.0;
            imGray1[i][j] = 0.0;
            dispMapL[i][j] = 0;
            dispMapR[i][j] = 0;
            dispMap[i][j] = 0;
            dispMapF[i][j] = 0;
        }
    }
}

void resizeAndGrayScale(){
    
    
    // Pixel Index ( y * width + x )
    // R = 4 * y * width + 4 * x + 0
    // G = 4 * y * width + 4 * x + 1
    // B = 4 * y * width + 4 * x + 2
    // A = 4 * y * width + 4 * x + 3
    
    int rInd, gInd, bInd;
    int x = winX + dMax - 1, y = winY - 1;
    for(int i = 0; i < imW; i += 4){
        for(int j = 0; j < imH; j += 4 ){
            
            rInd = 4 * j * imW + 4 * i;
            gInd = 4 * j * imW + 4 * i + 1;
            bInd = 4 * j * imW + 4 * i + 2;
            
            imGray0[x][y] = RC * im0[rInd] +  GC * im0[gInd] + BC * im0[bInd];
            imGray1[x][y] = RC * im1[rInd] +  GC * im1[gInd] + BC * im1[bInd];
            
            y++;
        }
        x++;
        y = winX - 1;
    }
}

void computeDisparityMap(){
    
    for (int x = winX + dMax - 1 ; x < mapW - dMax - winX - 1; x++){
        for (int y = winY - 1 ; y < mapH - winY - 1; y++ ){
            
            float maxSumL = 0;
            float maxSumR = 0;
            int bestDispL = dMin;
            int bestDispR = dMin;
            
            for (int d = dMin; d <= dMax; d++){
                
                float meanLL = 0, meanLR = 0, meanRL = 0, meanRR = 0;
                
                for (int j = 0; j < winY  ; j++ ){
                    for (int i = 0; i < winX; i++){
                        
                        meanLL += imGray1[x + i][y + j];
                        meanLR += imGray0[x + i + d][y + j];
                        
                        meanRL += imGray1[x + i - d][y + j];
                        meanRR += imGray0[x + i][y + j];
                    }
                }
                
                meanLL /= (winX * winY);
                meanLR /= (winX * winY);
                meanRL /= (winX * winY);
                meanRR /= (winX * winY);
                
                float left,right;
                float numL = 0, numR = 0, demLL = 0, demLR = 0, demRL = 0, demRR = 0;
                
                // Calculate ZNCC value for each window
                for (int j = 0; j < winY ; j++ ){
                    for (int i = 0; i < winX; i++){
                        
                        
                        left = imGray1[x + i][y + j];
                        right = imGray0[x + i + d][y + j];
                        
                        numL += ((left - meanLL) * (right - meanLR));
                        demLL += ((left - meanLL) * (left - meanLL));
                        demLR += ((right - meanLR) * (right - meanLR));
                        
                        left = imGray1[x + i - d][y + j];
                        right = imGray0[x + i][y + j];
                        
                        numR += ((right - meanRR) * (left - meanRL));
                        demRL += ((left - meanRL) * (left - meanRL));
                        demRR += ((right - meanRR) * (right - meanRR));
                        
                    }
                }
                
                float windowSumL = numL / sqrt(demLL * demLR);
                float windowSumR = numR / sqrt(demRL * demRR);
                
                // printf("SumL : %f, d : %d\n",windowSumL,d);
                // printf("SumR : %f, d : %d\n",windowSumR,d);
                
                if ( windowSumL > maxSumL ){
                    maxSumL = windowSumL;
                    bestDispL = d;
                }
                
                if ( windowSumR > maxSumR ){
                    maxSumR = windowSumR;
                    bestDispR = d;
                }
            }
            
            dispMapR[x][y] = bestDispR;
            dispMapL[x][y] = bestDispL;
            //printf("(%d,%d) , - R : %d , L : %d \n",x,y,bestDispR,bestDispL);
            
        }
    }
}

// Cross Check Post Processing Implementation
void crossCheck(){
    
    
    for (int x = winX + dMax - 1 ; x < mapW - dMax - winX - 1; x++){
        for (int y = winY - 1 ; y < mapH - winY - 1; y++ ){
            
            int x1 = x - dispMapR[x][y];
            
            if (x1 > 0){
                if ( abs(dispMapL[x1][y] - dispMapR[x][y]) <= threshold ){
                    dispMap[x][y] =  dispMapR[x][y];
                }
            }
            
        }
    }
    
    
    
}

// Oclusion Filling Post Processing Implementation
void doOcclusionFilling(){
    
    int resetBarr = 100;
    
    for (int x = winX + dMax - 1  ; x < mapW - dMax - winX - 1; x++){
        for (int y = winY - 1; y < mapH - winY - 1; y++){
            
            if (dispMap[x][y] == 0){
                //printf("0 Detected ... (%d,%d)\n",(int)x,(int)y);
                int counter = 0;
                
                do {
                    counter++;
                    for(int i = x - counter; i <  x + counter; i++ ){
                        for(int j = y - counter; j < y + counter; j++){
                            
                            if(dispMap[i][j] != 0){
                                dispMapF[x][y] = dispMap[i][j];
                                
                                counter = resetBarr + 1;
                                i = x + counter;
                                j = y + counter;
                               // printf("Found... (%d,%d)\n",(int)x,(int)y);
                            }
                            
                        }
                    }
                    
                } while(counter <= resetBarr);
                //printf("Bye... (%d,%d)\n",(int)x,(int)y);
            }else {
                dispMapF[x][y] = dispMap[x][y];
            }
        }
    }
    
}

int **normalizeImage(int **image, int maxPixelValue){
    
    int **out = new int *[mapW];
    for (int i = 0; i < mapW; i++){
        out[i] = new int [mapH];
    }
    
    double max = 65;
    double min = 0;
    
    // Normalize the value between [0 1] and then multiply it by max Pixel Value
    for(int i = 0; i < mapW; i++){
        for(int j = 0; j < mapH; j++){
            double result = ((double)image[i][j] - min ) / (max - min);
            result *= maxPixelValue;
            out[i][j] = result;
        }
    }
    return out;
}

void outputGrayScaleImage(float **image, const char *filename){
    
    // Encoding from 8 bit gray scale image to 32 bits RGBA raw pixels
    vector<unsigned char> grayImage;
    grayImage.resize(mapW * mapH * 1);
    
    for (int i = 0; i < mapW; i++){
        for(int j = 0; j < mapH; j++){
            grayImage[mapW * j + i] = image[i][j];
        }
    }
    
    vector<unsigned char> png;
    unsigned error = lodepng::encode(png, grayImage, (unsigned) mapW, (unsigned) mapH, LodePNGColorType::LCT_GREY , 8);
    if(!error) lodepng::save_file(png, filename);
    
    if(error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;
}

void outputImage(int **image, const char *filename){
    
    // Encoding from 8 bit gray scale image to 32 bits RGBA raw pixels
    int offset = 8;
    int imW = rW + offset * 2;
    int imH = rH + offset * 2;
    vector<unsigned char> grayImage;
    grayImage.resize(imW * imH * 1);
    int x = 0, y = 0;
    
    for (int i = winX + dMax - 1 - offset ; i < mapW - dMax - winX + 1 + offset ; i++){
        for (int j = winY - 1 - offset ; j < mapH - winY + 1 + offset; j++ ){
            grayImage[imW * x + y] = (int)image[i][j];
            x++;
        }
        x = 0;
        y++;
    }
    
    vector<unsigned char> png;
    unsigned error = lodepng::encode(png, grayImage, (unsigned) imW, (unsigned) imH, LodePNGColorType::LCT_GREY , 8);
    if(!error) lodepng::save_file(png, filename);
    
    if(error) cout << "encoder error " << error << ": "<< lodepng_error_text(error) << endl;
}

void write2DIntarray2file(const char *filename,int **data){
    ofstream file;
    file.open(filename);
    for (int i = 0; i < mapW; i++){
        for (int j = 0; j < mapH; j++){
            file << data[j][i] << " ";
        }
        file << endl;
    }
    file.close();
}

void write2DDoublearray2file(const char *filename,float **data){
    ofstream file;
    file.open(filename);
    for (int i = 0; i < mapH; i++){
        for (int j = 0; j < mapW; j++){
            file << data[j][i] << " ";
        }
        file << endl;
    }
    file.close();
}

